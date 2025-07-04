use std::ffi::CString;
use std::sync::Arc;

use rustc_data_structures::memmap::Mmap;
use rustc_errors::{DiagCtxtHandle, FatalError};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::middle::exported_symbols::{SymbolExportInfo, SymbolExportLevel};
use rustc_session::config::{CrateType, Lto};
use tracing::info;

use crate::back::symbol_export;
use crate::back::write::CodegenContext;
use crate::errors::{DynamicLinkingWithLTO, LtoDisallowed, LtoDylib, LtoProcMacro};
use crate::traits::*;

pub struct ThinModule<B: WriteBackendMethods> {
    pub shared: Arc<ThinShared<B>>,
    pub idx: usize,
}

impl<B: WriteBackendMethods> ThinModule<B> {
    pub fn name(&self) -> &str {
        self.shared.module_names[self.idx].to_str().unwrap()
    }

    pub fn cost(&self) -> u64 {
        // Yes, that's correct, we're using the size of the bytecode as an
        // indicator for how costly this codegen unit is.
        self.data().len() as u64
    }

    pub fn data(&self) -> &[u8] {
        let a = self.shared.thin_buffers.get(self.idx).map(|b| b.data());
        a.unwrap_or_else(|| {
            let len = self.shared.thin_buffers.len();
            self.shared.serialized_modules[self.idx - len].data()
        })
    }
}

pub struct ThinShared<B: WriteBackendMethods> {
    pub data: B::ThinData,
    pub thin_buffers: Vec<B::ThinBuffer>,
    pub serialized_modules: Vec<SerializedModule<B::ModuleBuffer>>,
    pub module_names: Vec<CString>,
}

pub enum SerializedModule<M: ModuleBufferMethods> {
    Local(M),
    FromRlib(Vec<u8>),
    FromUncompressedFile(Mmap),
}

impl<M: ModuleBufferMethods> SerializedModule<M> {
    pub fn data(&self) -> &[u8] {
        match *self {
            SerializedModule::Local(ref m) => m.data(),
            SerializedModule::FromRlib(ref m) => m,
            SerializedModule::FromUncompressedFile(ref m) => m,
        }
    }
}

fn crate_type_allows_lto(crate_type: CrateType) -> bool {
    match crate_type {
        CrateType::Executable
        | CrateType::Dylib
        | CrateType::Staticlib
        | CrateType::Cdylib
        | CrateType::ProcMacro
        | CrateType::Sdylib => true,
        CrateType::Rlib => false,
    }
}

pub fn exported_symbols_for_lto<'a, B: WriteBackendMethods>(
    cgcx: &'a CodegenContext<B>,
    dcx: DiagCtxtHandle<'_>,
) -> Result<Vec<&'a str>, FatalError> {
    // FIXME move symbol filtering to cg_ssa
    let export_threshold = match cgcx.lto {
        // We're just doing LTO for our one crate
        Lto::ThinLocal => SymbolExportLevel::Rust,

        // We're doing LTO for the entire crate graph
        Lto::Fat | Lto::Thin => symbol_export::crates_export_threshold(&cgcx.crate_types),

        Lto::No => panic!("didn't request LTO but we're doing LTO"),
    };

    let symbol_filter = &|&(ref name, info): &'a (String, SymbolExportInfo)| {
        if info.level.is_below_threshold(export_threshold) || info.used {
            Some(name.as_str())
        } else {
            None
        }
    };
    let exported_symbols = cgcx.exported_symbols.as_ref().expect("needs exported symbols for LTO");
    let mut symbols_below_threshold = {
        let _timer = cgcx.prof.generic_activity("lto_generate_symbols_below_threshold");
        exported_symbols[&LOCAL_CRATE].iter().filter_map(symbol_filter).collect::<Vec<&str>>()
    };
    info!("{} symbols to preserve in this crate", symbols_below_threshold.len());

    // If we're performing LTO for the entire crate graph, then for each of our
    // upstream dependencies, include their exported symbols.
    if cgcx.lto != Lto::ThinLocal {
        // Make sure we actually can run LTO
        for crate_type in cgcx.crate_types.iter() {
            if !crate_type_allows_lto(*crate_type) {
                return Err(dcx.emit_almost_fatal(LtoDisallowed));
            } else if *crate_type == CrateType::Dylib {
                if !cgcx.opts.unstable_opts.dylib_lto {
                    return Err(dcx.emit_almost_fatal(LtoDylib));
                }
            } else if *crate_type == CrateType::ProcMacro && !cgcx.opts.unstable_opts.dylib_lto {
                return Err(dcx.emit_almost_fatal(LtoProcMacro));
            }
        }

        if cgcx.opts.cg.prefer_dynamic && !cgcx.opts.unstable_opts.dylib_lto {
            return Err(dcx.emit_almost_fatal(DynamicLinkingWithLTO));
        }

        for &(cnum, ref _path) in cgcx.each_linked_rlib_for_lto.iter() {
            let exported_symbols =
                cgcx.exported_symbols.as_ref().expect("needs exported symbols for LTO");
            {
                let _timer = cgcx.prof.generic_activity("lto_generate_symbols_below_threshold");
                symbols_below_threshold
                    .extend(exported_symbols[&cnum].iter().filter_map(symbol_filter));
            }
        }
    }

    Ok(symbols_below_threshold)
}
