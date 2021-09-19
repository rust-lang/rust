use crate::attributes;
use crate::base;
use crate::context::CodegenCx;
use crate::llvm;
use crate::type_of::LayoutLlvmExt;
use rustc_codegen_ssa::traits::*;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
pub use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_middle::ty::layout::{FnAbiOf, LayoutOf};
use rustc_middle::ty::{self, Instance, TypeFoldable};
use rustc_session::config::CrateType;
use rustc_target::spec::RelocModel;
use tracing::debug;

impl PreDefineMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn predefine_static(
        &self,
        def_id: DefId,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    ) {
        let instance = Instance::mono(self.tcx, def_id);
        let ty = instance.ty(self.tcx, ty::ParamEnv::reveal_all());
        let llty = self.layout_of(ty).llvm_type(self);

        let g = self.define_global(symbol_name, llty).unwrap_or_else(|| {
            self.sess().span_fatal(
                self.tcx.def_span(def_id),
                &format!("symbol `{}` is already defined", symbol_name),
            )
        });

        unsafe {
            llvm::LLVMRustSetLinkage(g, base::linkage_to_llvm(linkage));
            llvm::LLVMRustSetVisibility(g, base::visibility_to_llvm(visibility));
            if self.should_assume_dso_local(g, false) {
                llvm::LLVMRustSetDSOLocal(g, true);
            }
        }

        self.instances.borrow_mut().insert(instance, g);
    }

    fn predefine_fn(
        &self,
        instance: Instance<'tcx>,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    ) {
        assert!(!instance.substs.needs_infer());

        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());
        let lldecl = self.declare_fn(symbol_name, &fn_abi);
        unsafe { llvm::LLVMRustSetLinkage(lldecl, base::linkage_to_llvm(linkage)) };
        let attrs = self.tcx.codegen_fn_attrs(instance.def_id());
        base::set_link_section(lldecl, &attrs);
        if linkage == Linkage::LinkOnceODR || linkage == Linkage::WeakODR {
            llvm::SetUniqueComdat(self.llmod, lldecl);
        }

        // If we're compiling the compiler-builtins crate, e.g., the equivalent of
        // compiler-rt, then we want to implicitly compile everything with hidden
        // visibility as we're going to link this object all over the place but
        // don't want the symbols to get exported.
        if linkage != Linkage::Internal
            && linkage != Linkage::Private
            && self.tcx.is_compiler_builtins(LOCAL_CRATE)
        {
            unsafe {
                llvm::LLVMRustSetVisibility(lldecl, llvm::Visibility::Hidden);
            }
        } else {
            unsafe {
                llvm::LLVMRustSetVisibility(lldecl, base::visibility_to_llvm(visibility));
            }
        }

        debug!("predefine_fn: instance = {:?}", instance);

        attributes::from_fn_attrs(self, lldecl, instance);

        unsafe {
            if self.should_assume_dso_local(lldecl, false) {
                llvm::LLVMRustSetDSOLocal(lldecl, true);
            }
        }

        self.instances.borrow_mut().insert(instance, lldecl);
    }
}

impl CodegenCx<'ll, 'tcx> {
    /// Whether a definition or declaration can be assumed to be local to a group of
    /// libraries that form a single DSO or executable.
    pub(crate) unsafe fn should_assume_dso_local(
        &self,
        llval: &llvm::Value,
        is_declaration: bool,
    ) -> bool {
        let linkage = llvm::LLVMRustGetLinkage(llval);
        let visibility = llvm::LLVMRustGetVisibility(llval);

        if matches!(linkage, llvm::Linkage::InternalLinkage | llvm::Linkage::PrivateLinkage) {
            return true;
        }

        if visibility != llvm::Visibility::Default && linkage != llvm::Linkage::ExternalWeakLinkage
        {
            return true;
        }

        // Symbols from executables can't really be imported any further.
        let all_exe = self.tcx.sess.crate_types().iter().all(|ty| *ty == CrateType::Executable);
        let is_declaration_for_linker =
            is_declaration || linkage == llvm::Linkage::AvailableExternallyLinkage;
        if all_exe && !is_declaration_for_linker {
            return true;
        }

        // PowerPC64 prefers TOC indirection to avoid copy relocations.
        if matches!(&*self.tcx.sess.target.arch, "powerpc64" | "powerpc64le") {
            return false;
        }

        // Thread-local variables generally don't support copy relocations.
        let is_thread_local_var = llvm::LLVMIsAGlobalVariable(llval)
            .map(|v| llvm::LLVMIsThreadLocal(v) == llvm::True)
            .unwrap_or(false);
        if is_thread_local_var {
            return false;
        }

        // Match clang by only supporting COFF and ELF for now.
        if self.tcx.sess.target.is_like_osx {
            return false;
        }

        // Static relocation model should force copy relocations everywhere.
        if self.tcx.sess.relocation_model() == RelocModel::Static {
            return true;
        }

        return false;
    }
}
