use std::borrow::Cow;
use std::ffi::CString;

use rustc_abi::AddressSpace;
use rustc_codegen_ssa::traits::*;
use rustc_hir::attrs::Linkage;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::mir::mono::Visibility;
use rustc_middle::ty::layout::{FnAbiOf, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{self, Instance, Ty, TypeVisitableExt};
use rustc_session::config::CrateType;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::{Arch, RelocModel};
use tracing::debug;

use crate::abi::FnAbiLlvmExt;
use crate::builder::Builder;
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

        let attrs = self.tcx.codegen_instance_attrs(instance.def);
        self.add_static_aliases(g, &attrs.foreign_item_symbol_aliases);

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

        let attrs = self.tcx.codegen_instance_attrs(instance.def);

        let lldecl =
            self.predefine_without_aliases(instance, &attrs, linkage, visibility, symbol_name);
        self.add_function_aliases(instance, lldecl, &attrs, &attrs.foreign_item_symbol_aliases);

        self.instances.borrow_mut().insert(instance, lldecl);
    }
}

impl<'ll, 'tcx> CodegenCx<'ll, 'tcx> {
    fn predefine_without_aliases(
        &self,
        instance: Instance<'tcx>,
        attrs: &Cow<'_, CodegenFnAttrs>,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    ) -> &'ll llvm::Value {
        let fn_abi: &FnAbi<'tcx, Ty<'tcx>> = self.fn_abi_of_instance(instance, ty::List::empty());
        let lldecl = self.declare_fn(symbol_name, fn_abi, Some(instance));
        llvm::set_linkage(lldecl, base::linkage_to_llvm(linkage));
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

        lldecl
    }

    /// LLVM has the concept of an `alias`.
    /// We need this for the "externally implementable items" feature,
    /// though it's generally useful.
    ///
    /// On macos, though this might be a more general problem, function symbols
    /// have a fixed target architecture. This is necessary, since macos binaries
    /// may contain code for both ARM and x86 macs.
    ///
    /// LLVM *can* add attributes for target architecture to function symbols,
    /// cannot do so for statics, but importantly, also cannot for aliases
    /// *even* when aliases may refer to a function symbol.
    ///
    /// This is not a problem: instead of using LLVM aliases, we can just generate
    /// a new function symbol (with target architecture!) which effectively comes down to:
    ///
    /// ```rust,ignore
    /// fn alias_name(...args) {
    ///     original_name(...args)
    /// }
    /// ```
    ///
    /// That's also an alias.
    ///
    /// This does mean that the alias symbol has a different address than the original symbol
    /// (assuming no optimizations by LLVM occur). This is unacceptable for statics.
    /// So for statics we do want to use LLVM aliases, which is fine,
    /// since for those we don't care about target architecture anyway.
    ///
    /// So, this function is for static aliases. See [`add_function_aliases`](Self::add_function_aliases) for the alternative.
    fn add_static_aliases(&self, aliasee: &llvm::Value, aliases: &[(DefId, Linkage, Visibility)]) {
        let ty = self.get_type_of_global(aliasee);

        for (alias, linkage, visibility) in aliases {
            let symbol_name = self.tcx.symbol_name(Instance::mono(self.tcx, *alias));
            tracing::debug!("STATIC ALIAS: {alias:?} {linkage:?} {visibility:?}");

            let lldecl = llvm::add_alias(
                self.llmod,
                ty,
                AddressSpace::ZERO,
                aliasee,
                &CString::new(symbol_name.name).unwrap(),
            );

            llvm::set_visibility(lldecl, base::visibility_to_llvm(*visibility));
            llvm::set_linkage(lldecl, base::linkage_to_llvm(*linkage));
        }
    }

    /// See [`add_static_aliases`](Self::add_static_aliases) for docs.
    fn add_function_aliases(
        &self,
        aliasee_instance: Instance<'tcx>,
        aliasee: &'ll llvm::Value,
        attrs: &Cow<'_, CodegenFnAttrs>,
        aliases: &[(DefId, Linkage, Visibility)],
    ) {
        for (alias, linkage, visibility) in aliases {
            let symbol_name = self.tcx.symbol_name(Instance::mono(self.tcx, *alias));
            tracing::debug!("FUNCTION ALIAS: {alias:?} {linkage:?} {visibility:?}");

            // predefine another copy of the original instance
            // with a new symbol name
            let alias_lldecl = self.predefine_without_aliases(
                aliasee_instance,
                attrs,
                *linkage,
                *visibility,
                symbol_name.name,
            );

            let fn_abi: &FnAbi<'tcx, Ty<'tcx>> =
                self.fn_abi_of_instance(aliasee_instance, ty::List::empty());

            // both the alias and the aliasee have the same ty
            let fn_ty = fn_abi.llvm_type(self);
            let start_llbb = Builder::append_block(self, alias_lldecl, "start");
            let mut start_bx = Builder::build(self, start_llbb);

            let num_params = llvm::count_params(alias_lldecl);
            let mut args = Vec::with_capacity(num_params as usize);
            for index in 0..num_params {
                args.push(llvm::get_param(alias_lldecl, index));
            }

            start_bx.tail_call(
                fn_ty,
                Some(attrs),
                fn_abi,
                aliasee,
                &args,
                None,
                Some(aliasee_instance),
            );
        }
    }

    /// A definition or declaration can be assumed to be local to a group of
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
        if self.tcx.sess.target.arch == Arch::PowerPC64 {
            return false;
        }

        // Match clang by only supporting COFF and ELF for now.
        if self.tcx.sess.target.is_like_darwin {
            return false;
        }

        // With pie relocation model, calls of functions defined in the translation
        // unit can use copy relocations.
        if self.tcx.sess.relocation_model() == RelocModel::Pie && !is_declaration {
            return true;
        }

        // Thread-local variables generally don't support copy relocations.
        let is_thread_local_var = llvm::LLVMIsAGlobalVariable(llval)
            .is_some_and(|v| llvm::LLVMIsThreadLocal(v).is_true());
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
