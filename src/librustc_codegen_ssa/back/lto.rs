use super::write::CodegenContext;
use crate::traits::*;
use crate::ModuleCodegen;

use rustc_errors::FatalError;

use std::sync::Arc;
use std::ffi::CString;

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


pub enum LtoModuleCodegen<B: WriteBackendMethods> {
    Fat {
        module: Option<ModuleCodegen<B::Module>>,
        _serialized_bitcode: Vec<SerializedModule<B::ModuleBuffer>>,
    },

    Thin(ThinModule<B>),
}

impl<B: WriteBackendMethods> LtoModuleCodegen<B> {
    pub fn name(&self) -> &str {
        match *self {
            LtoModuleCodegen::Fat { .. } => "everything",
            LtoModuleCodegen::Thin(ref m) => m.name(),
        }
    }

    /// Optimize this module within the given codegen context.
    ///
    /// This function is unsafe as it'll return a `ModuleCodegen` still
    /// points to LLVM data structures owned by this `LtoModuleCodegen`.
    /// It's intended that the module returned is immediately code generated and
    /// dropped, and then this LTO module is dropped.
    pub unsafe fn optimize(
        &mut self,
        cgcx: &CodegenContext<B>,
    ) -> Result<ModuleCodegen<B::Module>, FatalError> {
        match *self {
            LtoModuleCodegen::Fat { ref mut module, .. } => {
                let module = module.take().unwrap();
                {
                    let config = cgcx.config(module.kind);
                    B::run_lto_pass_manager(cgcx, &module, config, false);
                }
                Ok(module)
            }
            LtoModuleCodegen::Thin(ref mut thin) => B::optimize_thin(cgcx, thin),
        }
    }

    /// A "gauge" of how costly it is to optimize this module, used to sort
    /// biggest modules first.
    pub fn cost(&self) -> u64 {
        match *self {
            // Only one module with fat LTO, so the cost doesn't matter.
            LtoModuleCodegen::Fat { .. } => 0,
            LtoModuleCodegen::Thin(ref m) => m.cost(),
        }
    }
}


pub enum SerializedModule<M: ModuleBufferMethods> {
    Local(M),
    FromRlib(Vec<u8>),
    FromUncompressedFile(memmap::Mmap),
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
