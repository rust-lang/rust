use rustc_codegen_ssa::traits::*;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::bug;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_middle::ty::layout::{FnAbiOf, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{self, Instance, TypeVisitableExt};
use rustc_session::config::CrateType;
use rustc_target::spec::RelocModel;
use tracing::debug;

use crate::context::CodegenCx;
use crate::errors::SymbolAlreadyDefined;
use crate::type_of::LayoutLlvmExt;
use crate::{base, llvm};

impl<'tcx> PreDefineCodegenMethods<'tcx> for CodegenCx<'_, 'tcx> {
    fn predefine_static(
        &mut self,
        def_id: DefId,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    ) {
        let instance = Instance::mono(self.tcx, def_id);
        let DefKind::Static { nested, .. } = self.tcx.def_kind(def_id) else { bug!() };
        // Nested statics do not have a type, so pick a dummy type and let `codegen_static` figure
        // out the llvm type from the actual evaluated initializer.
        let ty =
            if nested { self.tcx.types.unit } else { instance.ty(self.tcx, self.typing_env()) };
        let llty = self.layout_of(ty).llvm_type(self);

        let g = self.define_global(symbol_name, llty).unwrap_or_else(|| {
            self.sess()
                .dcx()
                .emit_fatal(SymbolAlreadyDefined { span: self.tcx.def_span(def_id), symbol_name })
        });

        llvm::set_linkage(g, base::linkage_to_llvm(linkage));
        llvm::set_visibility(g, base::visibility_to_llvm(visibility));
        self.assume_dso_local(g, false);

        self.instances.borrow_mut().insert(instance, g);
    }

    fn predefine_fn(
        &mut self,
        instance: Instance<'tcx>,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    ) {
        assert!(!instance.args.has_infer());

        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());
        let lldecl = self.declare_fn(symbol_name, fn_abi, Some(instance));
        llvm::set_linkage(lldecl, base::linkage_to_llvm(linkage));
        let attrs = self.tcx.codegen_fn_attrs(instance.def_id());
        base::set_link_section(lldecl, attrs);
        if (linkage == Linkage::LinkOnceODR || linkage == Linkage::WeakODR)
            && self.tcx.sess.target.supports_comdat()
        {
            llvm::SetUniqueComdat(self.llmod, lldecl);
        }

        // If we're compiling the compiler-builtins crate, e.g., the equivalent of
        // compiler-rt, then we want to implicitly compile everything with hidden
        // visibility as we're going to link this object all over the place but
        // don't want the symbols to get exported.
        if linkage != Linkage::Internal && self.tcx.is_compiler_builtins(LOCAL_CRATE) {
            llvm::set_visibility(lldecl, llvm::Visibility::Hidden);
        } else {
            llvm::set_visibility(lldecl, base::visibility_to_llvm(visibility));
        }

        debug!("predefine_fn: instance = {:?}", instance);

        self.assume_dso_local(lldecl, false);

        self.instances.borrow_mut().insert(instance, lldecl);
    }
}

impl CodegenCx<'_, '_> {
    /// Whether a definition or declaration can be assumed to be local to a group of
    /// libraries that form a single DSO or executable.
    /// Marks the local as DSO if so.
    pub(crate) fn assume_dso_local(&self, llval: &llvm::Value, is_declaration: bool) -> bool {
        let assume = self.should_assume_dso_local(llval, is_declaration);
        if assume {
            llvm::set_dso_local(llval);
        }
        assume
    }

    fn should_assume_dso_local(&self, llval: &llvm::Value, is_declaration: bool) -> bool {
        let linkage = llvm::get_linkage(llval);
        let visibility = llvm::get_visibility(llval);

        if matches!(linkage, llvm::Linkage::InternalLinkage | llvm::Linkage::PrivateLinkage) {
            return true;
        }

        if visibility != llvm::Visibility::Default && linkage != llvm::Linkage::ExternalWeakLinkage
        {
            return true;
        }

        // Symbols from executables can't really be imported any further.
        let all_exe = self.tcx.crate_types().iter().all(|ty| *ty == CrateType::Executable);
        let is_declaration_for_linker =
            is_declaration || linkage == llvm::Linkage::AvailableExternallyLinkage;
        if all_exe && !is_declaration_for_linker {
            return true;
        }

        // PowerPC64 prefers TOC indirection to avoid copy relocations.
        if matches!(&*self.tcx.sess.target.arch, "powerpc64" | "powerpc64le") {
            return false;
        }

        // Match clang by only supporting COFF and ELF for now.
        if self.tcx.sess.target.is_like_darwin {
            return false;
        }

        // With pie relocation model calls of functions defined in the translation
        // unit can use copy relocations.
        if self.tcx.sess.relocation_model() == RelocModel::Pie && !is_declaration {
            return true;
        }

        // Thread-local variables generally don't support copy relocations.
        let is_thread_local_var = unsafe { llvm::LLVMIsAGlobalVariable(llval) }
            .is_some_and(|v| unsafe { llvm::LLVMIsThreadLocal(v) } == llvm::True);
        if is_thread_local_var {
            return false;
        }

        // Respect the direct-access-external-data to override default behavior if present.
        if let Some(direct) = self.tcx.sess.direct_access_external_data() {
            return direct;
        }

        // Static relocation model should force copy relocations everywhere.
        self.tcx.sess.relocation_model() == RelocModel::Static
    }
}
