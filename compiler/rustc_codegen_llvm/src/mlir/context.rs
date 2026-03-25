/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use core::borrow::Borrow;
use core::marker::PhantomData;
use core::mem::MaybeUninit;

use rustc_abi::Size;
use rustc_middle::mir::mono::CodegenUnit;
use rustc_middle::ty::TyCtxt;

use crate::context::{GenericCx, SCx};
use crate::llvm;
use crate::llvm::ffi::Type;
use crate::mlir::MlirModule;
use crate::mlir::ffi::{MLIRContext, ModuleOp};

/// MLIR-specific simple context.
/// Similar to the LLVM SCx but uses MLIR types.
pub(crate) struct MlirSCx<'ll> {
    pub llmod: &'ll ModuleOp,
    pub llcx: &'ll MLIRContext,
    pub isize_ty: &'ll Type,
    // Store an SCx for compatibility with GenericCx that requires Borrow<SCx>
    // This is a workaround since MLIR types can't be converted to LLVM types.
    // The SCx is created with dummy/null values and should not be used.
    scx: SCx<'ll>,
}

impl<'ll> Borrow<SCx<'ll>> for MlirSCx<'ll> {
    fn borrow(&self) -> &SCx<'ll> {
        &self.scx
    }
}

impl<'ll> Borrow<SCx<'ll>> for MlirFullCx<'ll, '_> {
    fn borrow(&self) -> &SCx<'ll> {
        // Borrow the SCx from the inner MlirSCx through GenericCx's Deref
        (*self.scx).borrow()
    }
}

impl<'ll> Borrow<MlirSCx<'ll>> for MlirFullCx<'ll, '_> {
    fn borrow(&self) -> &MlirSCx<'ll> {
        // Access the inner MlirSCx through GenericCx's Deref
        &*self.scx
    }
}

pub(crate) type MlirSimpleCx<'ll> = GenericCx<'ll, MlirSCx<'ll>>;

/// MLIR codegen context (one per codegen unit).
pub(crate) type MlirCodegenCx<'ll, 'tcx> = GenericCx<'ll, MlirFullCx<'ll, 'tcx>>;

pub(crate) struct MlirFullCx<'ll, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub scx: MlirSimpleCx<'ll>,
    pub codegen_unit: &'tcx CodegenUnit<'tcx>,
}

impl<'ll, 'tcx> MlirCodegenCx<'ll, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        codegen_unit: &'tcx CodegenUnit<'tcx>,
        llvm_module: &'ll MlirModule,
    ) -> Self {
        todo!()
        // let (llcx, llmod) = (&*llvm_module.llcx, llvm_module.llmod());

        // GenericCx(
        //     MlirFullCx {
        //         tcx,
        //         scx: MlirSimpleCx::new(llmod, llcx, tcx.data_layout.pointer_size()),
        //         codegen_unit,
        //     },
        //     PhantomData,
        // )
    }
}

impl<'ll> MlirSimpleCx<'ll> {
    pub(crate) fn new(llmod: &'ll ModuleOp, llcx: &'ll MLIRContext, pointer_size: Size) -> Self {
        // Create a dummy SCx for compatibility with GenericCx
        // Since MLIR types can't be converted to LLVM types, we use unsafe code
        // to create an uninitialized SCx. The SCx should not be accessed.
        // This is a workaround for the type system requirement.
        let dummy_scx = unsafe {
            // Use MaybeUninit to create an uninitialized SCx
            // This is unsafe but necessary for type system compatibility
            let mut uninit = MaybeUninit::<SCx<'ll>>::uninit();
            // Create dummy references - these are invalid but satisfy the type system
            // WARNING: These should never be dereferenced or used
            let dummy_module_ptr = 1usize as *const llvm::Module;
            let dummy_context_ptr = 1usize as *const llvm::Context;
            let dummy_type_ptr = 1usize as *const Type;
            let dummy_module: &'ll llvm::Module = &*dummy_module_ptr;
            let dummy_context: &'ll llvm::Context = &*dummy_context_ptr;
            let dummy_type: &'ll Type = &*dummy_type_ptr;
            uninit.write(SCx { llmod: dummy_module, llcx: dummy_context, isize_ty: dummy_type });
            uninit.assume_init()
        };

        // For isize_ty, we'll use the same dummy approach
        // In a real implementation, this would be created from MLIR context
        let dummy_isize_ty = unsafe {
            let dummy_type_ptr = 1usize as *const Type;
            &*dummy_type_ptr
        };

        let mlir_scx = MlirSCx { llmod, llcx, isize_ty: dummy_isize_ty, scx: dummy_scx };

        Self(mlir_scx, PhantomData)
    }
}
