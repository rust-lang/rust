use gccjit::{FunctionType, RValue};
use rustc_codegen_ssa::mir::debuginfo::{FunctionDebugContext, VariableKind};
use rustc_codegen_ssa::traits::{BuilderMethods, DebugInfoBuilderMethods, DebugInfoMethods};
use rustc_middle::middle::cstore::CrateDepKind;
use rustc_middle::mir;
use rustc_middle::ty::{Instance, Ty};
use rustc_span::{SourceFile, Span, Symbol};
use rustc_span::def_id::LOCAL_CRATE;
use rustc_target::abi::Size;
use rustc_target::abi::call::FnAbi;

use crate::builder::Builder;
use crate::context::CodegenCx;

impl<'a, 'gcc, 'tcx> DebugInfoBuilderMethods for Builder<'a, 'gcc, 'tcx> {
    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `dbg`, `debug`, `debuginfo`, `debug_info` etc.).
    fn dbg_var_addr(&mut self, _dbg_var: Self::DIVariable, _scope_metadata: Self::DIScope, _variable_alloca: Self::Value, _direct_offset: Size, _indirect_offsets: &[Size]) {
        unimplemented!();
        /*let cx = self.cx();

        // Convert the direct and indirect offsets to address ops.
        // FIXME(eddyb) use `const`s instead of getting the values via FFI,
        // the values should match the ones in the DWARF standard anyway.
        let op_deref = || unsafe { llvm::LLVMRustDIBuilderCreateOpDeref() };
        let op_plus_uconst = || unsafe { llvm::LLVMRustDIBuilderCreateOpPlusUconst() };
        let mut addr_ops = SmallVec::<[_; 8]>::new();

        if direct_offset.bytes() > 0 {
            addr_ops.push(op_plus_uconst());
            addr_ops.push(direct_offset.bytes() as i64);
        }
        for &offset in indirect_offsets {
            addr_ops.push(op_deref());
            if offset.bytes() > 0 {
                addr_ops.push(op_plus_uconst());
                addr_ops.push(offset.bytes() as i64);
            }
        }

        // FIXME(eddyb) maybe this information could be extracted from `dbg_var`,
        // to avoid having to pass it down in both places?
        // NB: `var` doesn't seem to know about the column, so that's a limitation.
        let dbg_loc = cx.create_debug_loc(scope_metadata, span);
        unsafe {
            // FIXME(eddyb) replace `llvm.dbg.declare` with `llvm.dbg.addr`.
            llvm::LLVMRustDIBuilderInsertDeclareAtEnd(
                DIB(cx),
                variable_alloca,
                dbg_var,
                addr_ops.as_ptr(),
                addr_ops.len() as c_uint,
                dbg_loc,
                self.llbb(),
            );
        }*/
    }

    /*fn set_source_location(&mut self, scope: Self::DIScope, span: Span) {
        unimplemented!();
        /*debug!("set_source_location: {}", self.sess().source_map().span_to_string(span));

        let dbg_loc = self.cx().create_debug_loc(scope, span);

        unsafe {
            llvm::LLVMSetCurrentDebugLocation(self.llbuilder, dbg_loc);
        }*/
    }*/

    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {
        // TODO: replace with gcc_jit_context_new_global_with_initializer() if it's added:
        // https://gcc.gnu.org/pipermail/jit/2020q3/001225.html
        //
        // Call the function to initialize global values here.
        // We assume this is only called for the main function.
        use std::iter;

        for crate_num in self.cx.tcx.all_crate_nums(()).iter().copied().chain(iter::once(LOCAL_CRATE)) {
            // FIXME: better way to find if a crate is of proc-macro type?
            if crate_num == LOCAL_CRATE || self.cx.tcx.dep_kind(crate_num) != CrateDepKind::MacrosOnly {
                // NOTE: proc-macro crates are not included in the executable, so don't call their
                // initialization routine.
                let initializer_name = format!("__gccGlobalCrateInit{}", self.cx.tcx.crate_name(crate_num));
                let codegen_init_func = self.context.new_function(None, FunctionType::Extern, self.context.new_type::<()>(), &[],
                initializer_name, false);
                self.llbb().add_eval(None, self.context.new_call(None, codegen_init_func, &[]));
            }
        }

        // TODO
        //gdb::insert_reference_to_gdb_debug_scripts_section_global(self)
    }

    fn set_var_name(&mut self, _value: RValue<'gcc>, _name: &str) {
        unimplemented!();
        // Avoid wasting time if LLVM value names aren't even enabled.
        /*if self.sess().fewer_names() {
            return;
        }

        // Only function parameters and instructions are local to a function,
        // don't change the name of anything else (e.g. globals).
        let param_or_inst = unsafe {
            llvm::LLVMIsAArgument(value).is_some() || llvm::LLVMIsAInstruction(value).is_some()
        };
        if !param_or_inst {
            return;
        }

        // Avoid replacing the name if it already exists.
        // While we could combine the names somehow, it'd
        // get noisy quick, and the usefulness is dubious.
        if llvm::get_value_name(value).is_empty() {
            llvm::set_value_name(value, name.as_bytes());
        }*/
    }

    fn set_dbg_loc(&mut self, _dbg_loc: Self::DILocation) {
        unimplemented!();
        /*unsafe {
            let dbg_loc_as_llval = llvm::LLVMRustMetadataAsValue(self.cx().llcx, dbg_loc);
            llvm::LLVMSetCurrentDebugLocation(self.llbuilder, dbg_loc_as_llval);
        }*/
    }
}

impl<'gcc, 'tcx> DebugInfoMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn create_vtable_metadata(&self, _ty: Ty<'tcx>, _vtable: Self::Value) {
        //metadata::create_vtable_metadata(self, ty, vtable)
    }

    fn create_function_debug_context(&self, _instance: Instance<'tcx>, _fn_abi: &FnAbi<'tcx, Ty<'tcx>>, _llfn: RValue<'gcc>, _mir: &mir::Body<'tcx>) -> Option<FunctionDebugContext<Self::DIScope, Self::DILocation>> {
        // TODO
        None
    }

    fn extend_scope_to_file(&self, _scope_metadata: Self::DIScope, _file: &SourceFile) -> Self::DIScope {
        unimplemented!();
    }

    fn debuginfo_finalize(&self) {
        //unimplemented!();
    }

    fn create_dbg_var(&self, _variable_name: Symbol, _variable_type: Ty<'tcx>, _scope_metadata: Self::DIScope, _variable_kind: VariableKind, _span: Span) -> Self::DIVariable {
        unimplemented!();
    }

    fn dbg_scope_fn(&self, _instance: Instance<'tcx>, _fn_abi: &FnAbi<'tcx, Ty<'tcx>>, _maybe_definition_llfn: Option<RValue<'gcc>>) -> Self::DIScope {
        unimplemented!();
        /*let def_id = instance.def_id();
        let containing_scope = get_containing_scope(self, instance);
        let span = self.tcx.def_span(def_id);
        let loc = self.lookup_debug_loc(span.lo());
        let file_metadata = file_metadata(self, &loc.file);

        let function_type_metadata = unsafe {
            let fn_signature = get_function_signature(self, fn_abi);
            llvm::LLVMRustDIBuilderCreateSubroutineType(DIB(self), fn_signature)
        };

        // Find the enclosing function, in case this is a closure.
        let def_key = self.tcx().def_key(def_id);
        let mut name = def_key.disambiguated_data.data.to_string();

        let enclosing_fn_def_id = self.tcx().closure_base_def_id(def_id);

        // Get_template_parameters() will append a `<...>` clause to the function
        // name if necessary.
        let generics = self.tcx().generics_of(enclosing_fn_def_id);
        let substs = instance.substs.truncate_to(self.tcx(), generics);
        let template_parameters = get_template_parameters(self, &generics, substs, &mut name);

        let linkage_name = &mangled_name_of_instance(self, instance).name;
        // Omit the linkage_name if it is the same as subprogram name.
        let linkage_name = if &name == linkage_name { "" } else { linkage_name };

        // FIXME(eddyb) does this need to be separate from `loc.line` for some reason?
        let scope_line = loc.line;

        let mut flags = DIFlags::FlagPrototyped;

        if fn_abi.ret.layout.abi.is_uninhabited() {
            flags |= DIFlags::FlagNoReturn;
        }

        let mut spflags = DISPFlags::SPFlagDefinition;
        if is_node_local_to_unit(self, def_id) {
            spflags |= DISPFlags::SPFlagLocalToUnit;
        }
        if self.sess().opts.optimize != config::OptLevel::No {
            spflags |= DISPFlags::SPFlagOptimized;
        }
        if let Some((id, _)) = self.tcx.entry_fn(LOCAL_CRATE) {
            if id.to_def_id() == def_id {
                spflags |= DISPFlags::SPFlagMainSubprogram;
            }
        }

        unsafe {
            return llvm::LLVMRustDIBuilderCreateFunction(
                DIB(self),
                containing_scope,
                name.as_ptr().cast(),
                name.len(),
                linkage_name.as_ptr().cast(),
                linkage_name.len(),
                file_metadata,
                loc.line.unwrap_or(UNKNOWN_LINE_NUMBER),
                function_type_metadata,
                scope_line.unwrap_or(UNKNOWN_LINE_NUMBER),
                flags,
                spflags,
                maybe_definition_llfn,
                template_parameters,
                None,
            );
        }

        fn get_function_signature<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        ) -> &'ll DIArray {
            if cx.sess().opts.debuginfo == DebugInfo::Limited {
                return create_DIArray(DIB(cx), &[]);
            }

            let mut signature = Vec::with_capacity(fn_abi.args.len() + 1);

            // Return type -- llvm::DIBuilder wants this at index 0
            signature.push(if fn_abi.ret.is_ignore() {
                None
            } else {
                Some(type_metadata(cx, fn_abi.ret.layout.ty, rustc_span::DUMMY_SP))
            });

            // Arguments types
            if cx.sess().target.options.is_like_msvc {
                // FIXME(#42800):
                // There is a bug in MSDIA that leads to a crash when it encounters
                // a fixed-size array of `u8` or something zero-sized in a
                // function-type (see #40477).
                // As a workaround, we replace those fixed-size arrays with a
                // pointer-type. So a function `fn foo(a: u8, b: [u8; 4])` would
                // appear as `fn foo(a: u8, b: *const u8)` in debuginfo,
                // and a function `fn bar(x: [(); 7])` as `fn bar(x: *const ())`.
                // This transformed type is wrong, but these function types are
                // already inaccurate due to ABI adjustments (see #42800).
                signature.extend(fn_abi.args.iter().map(|arg| {
                    let t = arg.layout.ty;
                    let t = match t.kind() {
                        ty::Array(ct, _)
                            if (*ct == cx.tcx.types.u8) || cx.layout_of(ct).is_zst() =>
                        {
                            cx.tcx.mk_imm_ptr(ct)
                        }
                        _ => t,
                    };
                    Some(type_metadata(cx, t, rustc_span::DUMMY_SP))
                }));
            } else {
                signature.extend(
                    fn_abi
                        .args
                        .iter()
                        .map(|arg| Some(type_metadata(cx, arg.layout.ty, rustc_span::DUMMY_SP))),
                );
            }

            create_DIArray(DIB(cx), &signature[..])
        }

        fn get_template_parameters<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            generics: &ty::Generics,
            substs: SubstsRef<'tcx>,
            name_to_append_suffix_to: &mut String,
        ) -> &'ll DIArray {
            if substs.types().next().is_none() {
                return create_DIArray(DIB(cx), &[]);
            }

            name_to_append_suffix_to.push('<');
            for (i, actual_type) in substs.types().enumerate() {
                if i != 0 {
                    name_to_append_suffix_to.push(',');
                }

                let actual_type =
                    cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), actual_type);
                // Add actual type name to <...> clause of function name
                let actual_type_name = compute_debuginfo_type_name(cx.tcx(), actual_type, true);
                name_to_append_suffix_to.push_str(&actual_type_name[..]);
            }
            name_to_append_suffix_to.push('>');

            // Again, only create type information if full debuginfo is enabled
            let template_params: Vec<_> = if cx.sess().opts.debuginfo == DebugInfo::Full {
                let names = get_parameter_names(cx, generics);
                substs
                    .iter()
                    .zip(names)
                    .filter_map(|(kind, name)| {
                        if let GenericArgKind::Type(ty) = kind.unpack() {
                            let actual_type =
                                cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), ty);
                            let actual_type_metadata =
                                type_metadata(cx, actual_type, rustc_span::DUMMY_SP);
                            let name = name.as_str();
                            Some(unsafe {
                                Some(llvm::LLVMRustDIBuilderCreateTemplateTypeParameter(
                                    DIB(cx),
                                    None,
                                    name.as_ptr().cast(),
                                    name.len(),
                                    actual_type_metadata,
                                ))
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                vec![]
            };

            create_DIArray(DIB(cx), &template_params[..])
        }

        fn get_parameter_names(cx: &CodegenCx<'_, '_>, generics: &ty::Generics) -> Vec<Symbol> {
            let mut names = generics
                .parent
                .map_or(vec![], |def_id| get_parameter_names(cx, cx.tcx.generics_of(def_id)));
            names.extend(generics.params.iter().map(|param| param.name));
            names
        }

        fn get_containing_scope<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            instance: Instance<'tcx>,
        ) -> &'ll DIScope {
            // First, let's see if this is a method within an inherent impl. Because
            // if yes, we want to make the result subroutine DIE a child of the
            // subroutine's self-type.
            let self_type = cx.tcx.impl_of_method(instance.def_id()).and_then(|impl_def_id| {
                // If the method does *not* belong to a trait, proceed
                if cx.tcx.trait_id_of_impl(impl_def_id).is_none() {
                    let impl_self_ty = cx.tcx.subst_and_normalize_erasing_regions(
                        instance.substs,
                        ty::ParamEnv::reveal_all(),
                        &cx.tcx.type_of(impl_def_id),
                    );

                    // Only "class" methods are generally understood by LLVM,
                    // so avoid methods on other types (e.g., `<*mut T>::null`).
                    match impl_self_ty.kind() {
                        ty::Adt(def, ..) if !def.is_box() => {
                            // Again, only create type information if full debuginfo is enabled
                            if cx.sess().opts.debuginfo == DebugInfo::Full
                                && !impl_self_ty.needs_subst()
                            {
                                Some(type_metadata(cx, impl_self_ty, rustc_span::DUMMY_SP))
                            } else {
                                Some(namespace::item_namespace(cx, def.did))
                            }
                        }
                        _ => None,
                    }
                } else {
                    // For trait method impls we still use the "parallel namespace"
                    // strategy
                    None
                }
            });

            self_type.unwrap_or_else(|| {
                namespace::item_namespace(
                    cx,
                    DefId {
                        krate: instance.def_id().krate,
                        index: cx
                            .tcx
                            .def_key(instance.def_id())
                            .parent
                            .expect("get_containing_scope: missing parent?"),
                    },
                )
            })
        }*/
    }

    fn dbg_loc(&self, _scope: Self::DIScope, _inlined_at: Option<Self::DILocation>, _span: Span) -> Self::DILocation {
        unimplemented!();
        /*let DebugLoc { line, col, .. } = self.lookup_debug_loc(span.lo());

        unsafe {
            llvm::LLVMRustDIBuilderCreateDebugLocation(
                utils::debug_context(self).llcontext,
                line.unwrap_or(UNKNOWN_LINE_NUMBER),
                col.unwrap_or(UNKNOWN_COLUMN_NUMBER),
                scope,
                inlined_at,
            )
        }*/
    }
}
