use std::ffi::CString;
use std::sync::Arc;

use rustc_data_structures::memmap::Mmap;
use rustc_hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc_middle::middle::exported_symbols::{ExportedSymbol, SymbolExportInfo, SymbolExportLevel};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{CrateType, Lto};
use tracing::info;

use crate::back::symbol_export::{self, allocator_shim_symbols, symbol_name_for_instance_in_crate};
use crate::back::write::CodegenContext;
use crate::base::allocator_kind_for_codegen;
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

pub(super) fn exported_symbols_for_lto(
    tcx: TyCtxt<'_>,
    each_linked_rlib_for_lto: &[CrateNum],
) -> Vec<String> {
    let export_threshold = match tcx.sess.lto() {
        // We're just doing LTO for our one crate
        Lto::ThinLocal => SymbolExportLevel::Rust,

        // We're doing LTO for the entire crate graph
        Lto::Fat | Lto::Thin => symbol_export::crates_export_threshold(&tcx.crate_types()),

        Lto::No => return vec![],
    };

    let copy_symbols = |cnum| {
        tcx.exported_non_generic_symbols(cnum)
            .iter()
            .chain(tcx.exported_generic_symbols(cnum))
            .filter_map(|&(s, info): &(ExportedSymbol<'_>, SymbolExportInfo)| {
                if info.level.is_below_threshold(export_threshold) || info.used {
                    Some(symbol_name_for_instance_in_crate(tcx, s, cnum))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };
    let mut symbols_below_threshold = {
        let _timer = tcx.prof.generic_activity("lto_generate_symbols_below_threshold");
        copy_symbols(LOCAL_CRATE)
    };
    info!("{} symbols to preserve in this crate", symbols_below_threshold.len());

    // If we're performing LTO for the entire crate graph, then for each of our
    // upstream dependencies, include their exported symbols.
    if tcx.sess.lto() != Lto::ThinLocal {
        for &cnum in each_linked_rlib_for_lto {
            let _timer = tcx.prof.generic_activity("lto_generate_symbols_below_threshold");
            symbols_below_threshold.extend(copy_symbols(cnum));
        }
    }

    // Mark allocator shim symbols as exported only if they were generated.
    if export_threshold == SymbolExportLevel::Rust && allocator_kind_for_codegen(tcx).is_some() {
        symbols_below_threshold.extend(allocator_shim_symbols(tcx).map(|(name, _kind)| name));
    }

    symbols_below_threshold
}

pub(super) fn check_lto_allowed<B: WriteBackendMethods>(cgcx: &CodegenContext<B>) {
    if cgcx.lto == Lto::ThinLocal {
        // Crate local LTO is always allowed
        return;
    }

    let dcx = cgcx.create_dcx();

    // Make sure we actually can run LTO
    for crate_type in cgcx.crate_types.iter() {
        if !crate_type_allows_lto(*crate_type) {
            dcx.handle().emit_fatal(LtoDisallowed);
        } else if *crate_type == CrateType::Dylib {
            if !cgcx.opts.unstable_opts.dylib_lto {
                dcx.handle().emit_fatal(LtoDylib);
            }
        } else if *crate_type == CrateType::ProcMacro && !cgcx.opts.unstable_opts.dylib_lto {
            dcx.handle().emit_fatal(LtoProcMacro);
        }
    }

    if cgcx.opts.cg.prefer_dynamic && !cgcx.opts.unstable_opts.dylib_lto {
        dcx.handle().emit_fatal(DynamicLinkingWithLTO);
    }
}
