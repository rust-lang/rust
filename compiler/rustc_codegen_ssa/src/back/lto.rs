use std::ffi::CString;
use std::sync::Arc;

use rustc_ast::expand::autodiff_attrs::AutoDiffItem;
use rustc_data_structures::memmap::Mmap;
use rustc_errors::FatalError;

use super::write::CodegenContext;
use crate::ModuleCodegen;
use crate::back::write::ModuleConfig;
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

pub enum LtoModuleCodegen<B: WriteBackendMethods> {
    Fat(ModuleCodegen<B::Module>),
    Thin(ThinModule<B>),
}

impl<B: WriteBackendMethods> LtoModuleCodegen<B> {
    pub fn name(&self) -> &str {
        match *self {
            LtoModuleCodegen::Fat(_) => "everything",
            LtoModuleCodegen::Thin(ref m) => m.name(),
        }
    }

    /// Optimize this module within the given codegen context.
    pub fn optimize(
        self,
        cgcx: &CodegenContext<B>,
    ) -> Result<ModuleCodegen<B::Module>, FatalError> {
        match self {
            LtoModuleCodegen::Fat(mut module) => {
                B::optimize_fat(cgcx, &mut module)?;
                Ok(module)
            }
            LtoModuleCodegen::Thin(thin) => B::optimize_thin(cgcx, thin),
        }
    }

    /// A "gauge" of how costly it is to optimize this module, used to sort
    /// biggest modules first.
    pub fn cost(&self) -> u64 {
        match *self {
            // Only one module with fat LTO, so the cost doesn't matter.
            LtoModuleCodegen::Fat(_) => 0,
            LtoModuleCodegen::Thin(ref m) => m.cost(),
        }
    }

    /// Run autodiff on Fat LTO module
    pub fn autodiff(
        self,
        cgcx: &CodegenContext<B>,
        diff_fncs: Vec<AutoDiffItem>,
        config: &ModuleConfig,
    ) -> Result<LtoModuleCodegen<B>, FatalError> {
        match &self {
            LtoModuleCodegen::Fat(module) => {
                B::autodiff(cgcx, &module, diff_fncs, config)?;
            }
            _ => panic!("autodiff called with non-fat LTO module"),
        }

        Ok(self)
    }
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
