use gccjit::Function;
#[cfg(feature = "master")]
use gccjit::{FnAttribute, LValue, ToRValue, VarAttribute};
use rustc_codegen_ssa::traits::PreDefineCodegenMethods;
use rustc_hir::attrs::Linkage;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mono::Visibility;
use rustc_middle::ty::layout::{FnAbiOf, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{self, Instance, TypeVisitableExt};

use crate::context::CodegenCx;
use crate::type_of::LayoutGccExt;
use crate::{attributes, base};

impl<'gcc, 'tcx> PreDefineCodegenMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    #[cfg_attr(not(feature = "master"), expect(unused_variables))]
    fn predefine_static(
        &mut self,
        def_id: DefId,
        _linkage: Linkage,
        visibility: Visibility,
        global_name: &str,
    ) {
        let attrs = self.tcx.codegen_fn_attrs(def_id);
        let instance = Instance::mono(self.tcx, def_id);
        let DefKind::Static { nested, .. } = self.tcx.def_kind(def_id) else { bug!() };
        // Nested statics do not have a type, so pick a dummy type and let `codegen_static` figure out
        // the gcc type from the actual evaluated initializer.
        let ty =
            if nested { self.tcx.types.unit } else { instance.ty(self.tcx, self.typing_env()) };
        let gcc_type = self.layout_of(ty).gcc_type(self);

        let is_tls = attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL);

        let create_global = |this: &CodegenCx<'gcc, 'tcx>, name: &str, visibility: Visibility| {
            let global = this.define_global(name, gcc_type, is_tls, attrs.link_section);
            #[cfg(feature = "master")]
            global.add_attribute(VarAttribute::Visibility(base::visibility_to_gcc(visibility)));
            // FIXME(antoyo): set linkage.
            global
        };
        let global = create_global(self, global_name, visibility);

        let attrs = self.tcx.codegen_instance_attrs(instance.def);
        #[cfg(feature = "master")]
        self.add_static_aliases(&attrs.foreign_item_symbol_aliases, global_name, &create_global);

        self.instances.borrow_mut().insert(instance, global);
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

        let decl =
            self.predefine_without_aliases(instance, &attrs, linkage, visibility, symbol_name);

        #[cfg(feature = "master")]
        self.add_function_aliases(instance, decl, &attrs, &attrs.foreign_item_symbol_aliases);

        self.functions.borrow_mut().insert(symbol_name.to_string(), decl);
        self.function_instances.borrow_mut().insert(instance, decl);
    }
}

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    #[cfg(feature = "master")]
    fn add_static_aliases<F>(
        &self,
        aliases: &[(DefId, Linkage, Visibility)],
        aliased: &str,
        create_global: &F,
    ) where
        F: Fn(&CodegenCx<'gcc, 'tcx>, &str, Visibility) -> LValue<'gcc>,
    {
        for &(alias, _linkage, visibility) in aliases {
            let instance = Instance::mono(self.tcx, alias);
            let symbol_name = self.tcx.symbol_name(instance);

            let alias = create_global(self, symbol_name.name, visibility);
            alias.add_attribute(VarAttribute::Alias(aliased));

            // Add the alias name to the set of cached items, so there is no duplicate
            // instance added to it during the normal `external static` codegen
            let prev_entry = self.instances.borrow_mut().insert(instance, alias);

            // If there already was a previous entry, then `add_static_aliases` was called multiple times for the same `alias`
            // which would result in incorrect codegen
            assert!(prev_entry.is_none(), "An instance was already present for {instance:?}");
        }
    }

    #[cfg(feature = "master")]
    fn add_function_aliases(
        &self,
        aliased_instance: Instance<'tcx>,
        aliased: Function<'gcc>,
        attrs: &CodegenFnAttrs,
        aliases: &[(DefId, Linkage, Visibility)],
    ) {
        for &(alias, linkage, visibility) in aliases {
            let symbol_name = self.tcx.symbol_name(Instance::mono(self.tcx, alias));

            // predefine another copy of the original instance
            // with a new symbol name
            let alias_fn_decl = self.predefine_without_aliases(
                aliased_instance,
                attrs,
                linkage,
                visibility,
                symbol_name.name,
            );

            let block = alias_fn_decl.new_block("start");
            let nb_params = alias_fn_decl.get_param_count();
            let mut args = Vec::with_capacity(nb_params);
            for idx in 0..nb_params {
                args.push(alias_fn_decl.get_param(idx as _).to_rvalue());
            }

            let void_type = self.context.new_type::<()>();
            let call = self.context.new_call(None, aliased, &args);
            if alias_fn_decl.get_return_type() == void_type {
                block.add_eval(None, call);
                block.end_with_void_return(None);
            } else {
                block.end_with_return(None, call);
            }
        }
    }

    fn predefine_without_aliases(
        &self,
        instance: Instance<'tcx>,
        _attrs: &CodegenFnAttrs,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    ) -> Function<'gcc> {
        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());
        self.linkage.set(base::linkage_to_gcc(linkage));
        let fn_decl = self.declare_fn(symbol_name, fn_abi);

        attributes::from_fn_attrs(self, fn_decl, instance);

        // If we're compiling the compiler-builtins crate, e.g., the equivalent of
        // compiler-rt, then we want to implicitly compile everything with hidden
        // visibility as we're going to link this object all over the place but
        // don't want the symbols to get exported.
        if linkage != Linkage::Internal && self.tcx.is_compiler_builtins(LOCAL_CRATE) {
            #[cfg(feature = "master")]
            fn_decl.add_attribute(FnAttribute::Visibility(gccjit::Visibility::Hidden));
        } else if visibility != Visibility::Default {
            #[cfg(feature = "master")]
            fn_decl.add_attribute(FnAttribute::Visibility(base::visibility_to_gcc(visibility)));
        }

        // FIXME(GuillaumeGomez): Add support for link section for `Function`.
        // fn_decl.set_link_section(&attrs.link_section);

        // FIXME(antoyo): set unique comdat.
        // FIXME(antoyo): use inline attribute from there in linkage.set() above.
        // FIXME: Should we handle dso?

        fn_decl
    }
}
