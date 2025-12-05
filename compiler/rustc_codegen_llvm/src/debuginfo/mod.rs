#![doc = include_str!("doc.md")]

use std::cell::{OnceCell, RefCell};
use std::ops::Range;
use std::sync::Arc;
use std::{iter, ptr};

use libc::c_uint;
use metadata::create_subroutine_type;
use rustc_abi::Size;
use rustc_codegen_ssa::debuginfo::type_names;
use rustc_codegen_ssa::mir::debuginfo::VariableKind::*;
use rustc_codegen_ssa::mir::debuginfo::{DebugScope, FunctionDebugContext, VariableKind};
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::unord::UnordMap;
use rustc_hir::def_id::{DefId, DefIdMap};
use rustc_index::IndexVec;
use rustc_middle::mir;
use rustc_middle::ty::layout::{HasTypingEnv, LayoutOf};
use rustc_middle::ty::{self, GenericArgsRef, Instance, Ty, TypeVisitableExt};
use rustc_session::Session;
use rustc_session::config::{self, DebugInfo};
use rustc_span::{
    BytePos, Pos, SourceFile, SourceFileAndLine, SourceFileHash, Span, StableSourceFileId, Symbol,
};
use rustc_target::callconv::FnAbi;
use rustc_target::spec::DebuginfoKind;
use smallvec::SmallVec;
use tracing::debug;

use self::create_scope_map::compute_mir_scopes;
pub(crate) use self::di_builder::DIBuilderExt;
pub(crate) use self::metadata::build_global_var_di_node;
use self::metadata::{
    UNKNOWN_COLUMN_NUMBER, UNKNOWN_LINE_NUMBER, file_metadata, spanned_type_di_node, type_di_node,
};
use self::namespace::mangled_name_of_instance;
use self::utils::{DIB, create_DIArray, is_node_local_to_unit};
use crate::builder::Builder;
use crate::common::{AsCCharPtr, CodegenCx};
use crate::llvm::debuginfo::{
    DIArray, DIBuilderBox, DIFile, DIFlags, DILexicalBlock, DILocation, DISPFlags, DIScope,
    DITemplateTypeParameter, DIType, DIVariable,
};
use crate::llvm::{self, Value};

mod create_scope_map;
mod di_builder;
mod dwarf_const;
mod gdb;
pub(crate) mod metadata;
mod namespace;
mod utils;

/// A context object for maintaining all state needed by the debuginfo module.
pub(crate) struct CodegenUnitDebugContext<'ll, 'tcx> {
    llmod: &'ll llvm::Module,
    builder: DIBuilderBox<'ll>,
    created_files: RefCell<UnordMap<Option<(StableSourceFileId, SourceFileHash)>, &'ll DIFile>>,

    type_map: metadata::TypeMap<'ll, 'tcx>,
    adt_stack: RefCell<Vec<(DefId, GenericArgsRef<'tcx>)>>,
    namespace_map: RefCell<DefIdMap<&'ll DIScope>>,
    recursion_marker_type: OnceCell<&'ll DIType>,
}

impl<'ll, 'tcx> CodegenUnitDebugContext<'ll, 'tcx> {
    pub(crate) fn new(llmod: &'ll llvm::Module) -> Self {
        debug!("CodegenUnitDebugContext::new");
        let builder = DIBuilderBox::new(llmod);
        // DIBuilder inherits context from the module, so we'd better use the same one
        CodegenUnitDebugContext {
            llmod,
            builder,
            created_files: Default::default(),
            type_map: Default::default(),
            adt_stack: Default::default(),
            namespace_map: RefCell::new(Default::default()),
            recursion_marker_type: OnceCell::new(),
        }
    }

    pub(crate) fn finalize(&self, sess: &Session) {
        unsafe { llvm::LLVMDIBuilderFinalize(self.builder.as_ref()) };

        match sess.target.debuginfo_kind {
            DebuginfoKind::Dwarf | DebuginfoKind::DwarfDsym => {
                // Debuginfo generation in LLVM by default uses a higher
                // version of dwarf than macOS currently understands. We can
                // instruct LLVM to emit an older version of dwarf, however,
                // for macOS to understand. For more info see #11352
                // This can be overridden using --llvm-opts -dwarf-version,N.
                // Android has the same issue (#22398)
                llvm::add_module_flag_u32(
                    self.llmod,
                    // In the case where multiple CGUs with different dwarf version
                    // values are being merged together, such as with cross-crate
                    // LTO, then we want to use the highest version of dwarf
                    // we can. This matches Clang's behavior as well.
                    llvm::ModuleFlagMergeBehavior::Max,
                    "Dwarf Version",
                    sess.dwarf_version(),
                );
            }
            DebuginfoKind::Pdb => {
                // Indicate that we want CodeView debug information
                llvm::add_module_flag_u32(
                    self.llmod,
                    llvm::ModuleFlagMergeBehavior::Warning,
                    "CodeView",
                    1,
                );
            }
        }

        // Prevent bitcode readers from deleting the debug info.
        llvm::add_module_flag_u32(
            self.llmod,
            llvm::ModuleFlagMergeBehavior::Warning,
            "Debug Info Version",
            unsafe { llvm::LLVMRustDebugMetadataVersion() },
        );
    }
}

/// Creates any deferred debug metadata nodes
pub(crate) fn finalize(cx: &CodegenCx<'_, '_>) {
    if let Some(dbg_cx) = &cx.dbg_cx {
        debug!("finalize");

        if gdb::needs_gdb_debug_scripts_section(cx) {
            // Add a .debug_gdb_scripts section to this compile-unit. This will
            // cause GDB to try and load the gdb_load_rust_pretty_printers.py file,
            // which activates the Rust pretty printers for binary this section is
            // contained in.
            gdb::get_or_insert_gdb_debug_scripts_section_global(cx);
        }

        dbg_cx.finalize(cx.sess());
    }
}

impl<'ll> Builder<'_, 'll, '_> {
    pub(crate) fn get_dbg_loc(&self) -> Option<&'ll DILocation> {
        unsafe { llvm::LLVMGetCurrentDebugLocation2(self.llbuilder) }
    }
}

impl<'ll, 'tcx> DebugInfoBuilderMethods<'tcx> for Builder<'_, 'll, 'tcx> {
    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `dbg`, `debug`, `debuginfo`, `debug_info` etc.).
    fn dbg_var_addr(
        &mut self,
        dbg_var: &'ll DIVariable,
        dbg_loc: &'ll DILocation,
        variable_alloca: Self::Value,
        direct_offset: Size,
        indirect_offsets: &[Size],
        fragment: &Option<Range<Size>>,
    ) {
        use dwarf_const::{DW_OP_LLVM_fragment, DW_OP_deref, DW_OP_plus_uconst};

        // Convert the direct and indirect offsets and fragment byte range to address ops.
        let mut addr_ops = SmallVec::<[u64; 8]>::new();

        if direct_offset.bytes() > 0 {
            addr_ops.push(DW_OP_plus_uconst);
            addr_ops.push(direct_offset.bytes());
        }
        for &offset in indirect_offsets {
            addr_ops.push(DW_OP_deref);
            if offset.bytes() > 0 {
                addr_ops.push(DW_OP_plus_uconst);
                addr_ops.push(offset.bytes());
            }
        }
        if let Some(fragment) = fragment {
            // `DW_OP_LLVM_fragment` takes as arguments the fragment's
            // offset and size, both of them in bits.
            addr_ops.push(DW_OP_LLVM_fragment);
            addr_ops.push(fragment.start.bits());
            addr_ops.push((fragment.end - fragment.start).bits());
        }

        let di_builder = DIB(self.cx());
        let addr_expr = di_builder.create_expression(&addr_ops);
        unsafe {
            llvm::LLVMDIBuilderInsertDeclareRecordAtEnd(
                di_builder,
                variable_alloca,
                dbg_var,
                addr_expr,
                dbg_loc,
                self.llbb(),
            )
        };
    }

    fn dbg_var_value(
        &mut self,
        dbg_var: &'ll DIVariable,
        dbg_loc: &'ll DILocation,
        value: Self::Value,
        direct_offset: Size,
        indirect_offsets: &[Size],
        fragment: &Option<Range<Size>>,
    ) {
        use dwarf_const::{DW_OP_LLVM_fragment, DW_OP_deref, DW_OP_plus_uconst, DW_OP_stack_value};

        // Convert the direct and indirect offsets and fragment byte range to address ops.
        let mut addr_ops = SmallVec::<[u64; 8]>::new();

        if direct_offset.bytes() > 0 {
            addr_ops.push(DW_OP_plus_uconst);
            addr_ops.push(direct_offset.bytes() as u64);
            addr_ops.push(DW_OP_stack_value);
        }
        for &offset in indirect_offsets {
            addr_ops.push(DW_OP_deref);
            if offset.bytes() > 0 {
                addr_ops.push(DW_OP_plus_uconst);
                addr_ops.push(offset.bytes() as u64);
            }
        }
        if let Some(fragment) = fragment {
            // `DW_OP_LLVM_fragment` takes as arguments the fragment's
            // offset and size, both of them in bits.
            addr_ops.push(DW_OP_LLVM_fragment);
            addr_ops.push(fragment.start.bits() as u64);
            addr_ops.push((fragment.end - fragment.start).bits() as u64);
        }

        let di_builder = DIB(self.cx());
        let addr_expr = unsafe {
            llvm::LLVMDIBuilderCreateExpression(di_builder, addr_ops.as_ptr(), addr_ops.len())
        };
        unsafe {
            llvm::LLVMDIBuilderInsertDbgValueRecordAtEnd(
                di_builder,
                value,
                dbg_var,
                addr_expr,
                dbg_loc,
                self.llbb(),
            );
        }
    }

    fn set_dbg_loc(&mut self, dbg_loc: &'ll DILocation) {
        unsafe {
            llvm::LLVMSetCurrentDebugLocation2(self.llbuilder, dbg_loc);
        }
    }

    fn clear_dbg_loc(&mut self) {
        unsafe {
            llvm::LLVMSetCurrentDebugLocation2(self.llbuilder, ptr::null());
        }
    }

    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {
        gdb::insert_reference_to_gdb_debug_scripts_section_global(self)
    }

    fn set_var_name(&mut self, value: &'ll Value, name: &str) {
        // Avoid wasting time if LLVM value names aren't even enabled.
        if self.sess().fewer_names() {
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
        }
    }

    /// Annotate move/copy operations with debug info for profiling.
    ///
    /// This creates a temporary debug scope that makes the move/copy appear as an inlined call to
    /// `compiler_move<T, SIZE>()` or `compiler_copy<T, SIZE>()`. The provided closure is executed
    /// with this temporary debug location active.
    ///
    /// The `instance` parameter should be the monomorphized instance of the `compiler_move` or
    /// `compiler_copy` function with the actual type and size.
    fn with_move_annotation<R>(
        &mut self,
        instance: ty::Instance<'tcx>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        // Save the current debug location
        let saved_loc = self.get_dbg_loc();

        // Create a DIScope for the compiler_move/compiler_copy function
        // We use the function's FnAbi for debug info generation
        let fn_abi = self
            .cx()
            .tcx
            .fn_abi_of_instance(
                self.cx().typing_env().as_query_input((instance, ty::List::empty())),
            )
            .unwrap();

        let di_scope = self.cx().dbg_scope_fn(instance, fn_abi, None);

        // Create an inlined debug location:
        // - scope: the compiler_move/compiler_copy function
        // - inlined_at: the current location (where the move/copy actually occurs)
        // - span: use the function's definition span
        let fn_span = self.cx().tcx.def_span(instance.def_id());
        let inlined_loc = self.cx().dbg_loc(di_scope, saved_loc, fn_span);

        // Set the temporary debug location
        self.set_dbg_loc(inlined_loc);

        // Execute the closure (which will generate the memcpy)
        let result = f(self);

        // Restore the original debug location
        if let Some(loc) = saved_loc {
            self.set_dbg_loc(loc);
        } else {
            self.clear_dbg_loc();
        }

        result
    }
}

/// A source code location used to generate debug information.
// FIXME(eddyb) rename this to better indicate it's a duplicate of
// `rustc_span::Loc` rather than `DILocation`, perhaps by making
// `lookup_char_pos` return the right information instead.
struct DebugLoc {
    /// Information about the original source file.
    file: Arc<SourceFile>,
    /// The (1-based) line number.
    line: u32,
    /// The (1-based) column number.
    col: u32,
}

impl<'ll> CodegenCx<'ll, '_> {
    /// Looks up debug source information about a `BytePos`.
    // FIXME(eddyb) rename this to better indicate it's a duplicate of
    // `lookup_char_pos` rather than `dbg_loc`, perhaps by making
    // `lookup_char_pos` return the right information instead.
    fn lookup_debug_loc(&self, pos: BytePos) -> DebugLoc {
        let (file, line, col) = match self.sess().source_map().lookup_line(pos) {
            Ok(SourceFileAndLine { sf: file, line }) => {
                let line_pos = file.lines()[line];

                // Use 1-based indexing.
                let line = (line + 1) as u32;
                let col = (file.relative_position(pos) - line_pos).to_u32() + 1;

                (file, line, col)
            }
            Err(file) => (file, UNKNOWN_LINE_NUMBER, UNKNOWN_COLUMN_NUMBER),
        };

        // For MSVC, omit the column number.
        // Otherwise, emit it. This mimics clang behaviour.
        // See discussion in https://github.com/rust-lang/rust/issues/42921
        if self.sess().target.is_like_msvc {
            DebugLoc { file, line, col: UNKNOWN_COLUMN_NUMBER }
        } else {
            DebugLoc { file, line, col }
        }
    }

    fn create_template_type_parameter(
        &self,
        name: &str,
        actual_type_metadata: &'ll DIType,
    ) -> &'ll DITemplateTypeParameter {
        unsafe {
            llvm::LLVMRustDIBuilderCreateTemplateTypeParameter(
                DIB(self),
                None,
                name.as_c_char_ptr(),
                name.len(),
                actual_type_metadata,
            )
        }
    }
}

impl<'ll, 'tcx> DebugInfoCodegenMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        llfn: &'ll Value,
        mir: &mir::Body<'tcx>,
    ) -> Option<FunctionDebugContext<'tcx, &'ll DIScope, &'ll DILocation>> {
        if self.sess().opts.debuginfo == DebugInfo::None {
            return None;
        }

        // Initialize fn debug context (including scopes).
        let empty_scope = DebugScope {
            dbg_scope: self.dbg_scope_fn(instance, fn_abi, Some(llfn)),
            inlined_at: None,
            file_start_pos: BytePos(0),
            file_end_pos: BytePos(0),
        };
        let mut fn_debug_context = FunctionDebugContext {
            scopes: IndexVec::from_elem(empty_scope, &mir.source_scopes),
            inlined_function_scopes: Default::default(),
        };

        // Fill in all the scopes, with the information from the MIR body.
        compute_mir_scopes(self, instance, mir, &mut fn_debug_context);

        Some(fn_debug_context)
    }

    fn dbg_scope_fn(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        maybe_definition_llfn: Option<&'ll Value>,
    ) -> &'ll DIScope {
        let tcx = self.tcx;

        let def_id = instance.def_id();
        let (containing_scope, is_method) = get_containing_scope(self, instance);
        let span = tcx.def_span(def_id);
        let loc = self.lookup_debug_loc(span.lo());
        let file_metadata = file_metadata(self, &loc.file);

        let function_type_metadata =
            create_subroutine_type(self, &get_function_signature(self, fn_abi));

        let mut name = String::with_capacity(64);
        type_names::push_item_name(tcx, def_id, false, &mut name);

        // Find the enclosing function, in case this is a closure.
        let enclosing_fn_def_id = tcx.typeck_root_def_id(def_id);

        // We look up the generics of the enclosing function and truncate the args
        // to their length in order to cut off extra stuff that might be in there for
        // closures or coroutines.
        let generics = tcx.generics_of(enclosing_fn_def_id);
        let args = instance.args.truncate_to(tcx, generics);

        type_names::push_generic_params(
            tcx,
            tcx.normalize_erasing_regions(self.typing_env(), args),
            &mut name,
        );

        let template_parameters = get_template_parameters(self, generics, args);

        let linkage_name = &mangled_name_of_instance(self, instance).name;
        // Omit the linkage_name if it is the same as subprogram name.
        let linkage_name = if &name == linkage_name { "" } else { linkage_name };

        // FIXME(eddyb) does this need to be separate from `loc.line` for some reason?
        let scope_line = loc.line;

        let mut flags = DIFlags::FlagPrototyped;

        if fn_abi.ret.layout.is_uninhabited() {
            flags |= DIFlags::FlagNoReturn;
        }

        let mut spflags = DISPFlags::SPFlagDefinition;
        if is_node_local_to_unit(self, def_id) {
            spflags |= DISPFlags::SPFlagLocalToUnit;
        }
        if self.sess().opts.optimize != config::OptLevel::No {
            spflags |= DISPFlags::SPFlagOptimized;
        }
        if let Some((id, _)) = tcx.entry_fn(()) {
            if id == def_id {
                spflags |= DISPFlags::SPFlagMainSubprogram;
            }
        }

        // When we're adding a method to a type DIE, we only want a DW_AT_declaration there, because
        // LLVM LTO can't unify type definitions when a child DIE is a full subprogram definition.
        // When we use this `decl` below, the subprogram definition gets created at the CU level
        // with a DW_AT_specification pointing back to the type's declaration.
        let decl = is_method.then(|| unsafe {
            llvm::LLVMRustDIBuilderCreateMethod(
                DIB(self),
                containing_scope,
                name.as_c_char_ptr(),
                name.len(),
                linkage_name.as_c_char_ptr(),
                linkage_name.len(),
                file_metadata,
                loc.line,
                function_type_metadata,
                flags,
                spflags & !DISPFlags::SPFlagDefinition,
                template_parameters,
            )
        });

        return unsafe {
            llvm::LLVMRustDIBuilderCreateFunction(
                DIB(self),
                containing_scope,
                name.as_c_char_ptr(),
                name.len(),
                linkage_name.as_c_char_ptr(),
                linkage_name.len(),
                file_metadata,
                loc.line,
                function_type_metadata,
                scope_line,
                flags,
                spflags,
                maybe_definition_llfn,
                template_parameters,
                decl,
            )
        };

        fn get_function_signature<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        ) -> Vec<Option<&'ll llvm::Metadata>> {
            if cx.sess().opts.debuginfo != DebugInfo::Full {
                return vec![];
            }

            let mut signature = Vec::with_capacity(fn_abi.args.len() + 1);

            // Return type -- llvm::DIBuilder wants this at index 0
            signature.push(if fn_abi.ret.is_ignore() {
                None
            } else {
                Some(type_di_node(cx, fn_abi.ret.layout.ty))
            });

            // Arguments types
            if cx.sess().target.is_like_msvc {
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
                            if (*ct == cx.tcx.types.u8) || cx.layout_of(*ct).is_zst() =>
                        {
                            Ty::new_imm_ptr(cx.tcx, *ct)
                        }
                        _ => t,
                    };
                    Some(type_di_node(cx, t))
                }));
            } else {
                signature
                    .extend(fn_abi.args.iter().map(|arg| Some(type_di_node(cx, arg.layout.ty))));
            }

            signature
        }

        fn get_template_parameters<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            generics: &ty::Generics,
            args: GenericArgsRef<'tcx>,
        ) -> &'ll DIArray {
            if args.types().next().is_none() {
                return create_DIArray(DIB(cx), &[]);
            }

            // Again, only create type information if full debuginfo is enabled
            let template_params: Vec<_> = if cx.sess().opts.debuginfo == DebugInfo::Full {
                let names = get_parameter_names(cx, generics);
                iter::zip(args, names)
                    .filter_map(|(kind, name)| {
                        kind.as_type().map(|ty| {
                            let actual_type = cx.tcx.normalize_erasing_regions(cx.typing_env(), ty);
                            let actual_type_metadata = type_di_node(cx, actual_type);
                            Some(cx.create_template_type_parameter(
                                name.as_str(),
                                actual_type_metadata,
                            ))
                        })
                    })
                    .collect()
            } else {
                vec![]
            };

            create_DIArray(DIB(cx), &template_params)
        }

        fn get_parameter_names(cx: &CodegenCx<'_, '_>, generics: &ty::Generics) -> Vec<Symbol> {
            let mut names = generics.parent.map_or_else(Vec::new, |def_id| {
                get_parameter_names(cx, cx.tcx.generics_of(def_id))
            });
            names.extend(generics.own_params.iter().map(|param| param.name));
            names
        }

        /// Returns a scope, plus `true` if that's a type scope for "class" methods,
        /// otherwise `false` for plain namespace scopes.
        fn get_containing_scope<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            instance: Instance<'tcx>,
        ) -> (&'ll DIScope, bool) {
            // First, let's see if this is a method within an inherent impl. Because
            // if yes, we want to make the result subroutine DIE a child of the
            // subroutine's self-type.
            // For trait method impls we still use the "parallel namespace"
            // strategy
            if let Some(imp_def_id) = cx.tcx.inherent_impl_of_assoc(instance.def_id()) {
                let impl_self_ty = cx.tcx.instantiate_and_normalize_erasing_regions(
                    instance.args,
                    cx.typing_env(),
                    cx.tcx.type_of(imp_def_id),
                );

                // Only "class" methods are generally understood by LLVM,
                // so avoid methods on other types (e.g., `<*mut T>::null`).
                if let ty::Adt(def, ..) = impl_self_ty.kind()
                    && !def.is_box()
                {
                    // Again, only create type information if full debuginfo is enabled
                    if cx.sess().opts.debuginfo == DebugInfo::Full && !impl_self_ty.has_param() {
                        return (type_di_node(cx, impl_self_ty), true);
                    } else {
                        return (namespace::item_namespace(cx, def.did()), false);
                    }
                }
            }

            let scope = namespace::item_namespace(
                cx,
                DefId {
                    krate: instance.def_id().krate,
                    index: cx
                        .tcx
                        .def_key(instance.def_id())
                        .parent
                        .expect("get_containing_scope: missing parent?"),
                },
            );
            (scope, false)
        }
    }

    fn dbg_loc(
        &self,
        scope: &'ll DIScope,
        inlined_at: Option<&'ll DILocation>,
        span: Span,
    ) -> &'ll DILocation {
        // When emitting debugging information, DWARF (i.e. everything but MSVC)
        // treats line 0 as a magic value meaning that the code could not be
        // attributed to any line in the source. That's also exactly what dummy
        // spans are. Make that equivalence here, rather than passing dummy spans
        // to lookup_debug_loc, which will return line 1 for them.
        let (line, col) = if span.is_dummy() && !self.sess().target.is_like_msvc {
            (0, 0)
        } else {
            let DebugLoc { line, col, .. } = self.lookup_debug_loc(span.lo());
            (line, col)
        };

        unsafe { llvm::LLVMDIBuilderCreateDebugLocation(self.llcx, line, col, scope, inlined_at) }
    }

    fn create_vtable_debuginfo(
        &self,
        ty: Ty<'tcx>,
        trait_ref: Option<ty::ExistentialTraitRef<'tcx>>,
        vtable: Self::Value,
    ) {
        metadata::create_vtable_di_node(self, ty, trait_ref, vtable)
    }

    fn extend_scope_to_file(
        &self,
        scope_metadata: &'ll DIScope,
        file: &rustc_span::SourceFile,
    ) -> &'ll DILexicalBlock {
        metadata::extend_scope_to_file(self, scope_metadata, file)
    }

    fn debuginfo_finalize(&self) {
        finalize(self)
    }

    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `dbg`, `debug`, `debuginfo`, `debug_info` etc.).
    fn create_dbg_var(
        &self,
        variable_name: Symbol,
        variable_type: Ty<'tcx>,
        scope_metadata: &'ll DIScope,
        variable_kind: VariableKind,
        span: Span,
    ) -> &'ll DIVariable {
        let loc = self.lookup_debug_loc(span.lo());
        let file_metadata = file_metadata(self, &loc.file);

        let type_metadata = spanned_type_di_node(self, variable_type, span);

        let align = self.align_of(variable_type);

        let name = variable_name.as_str();

        match variable_kind {
            ArgumentVariable(arg_index) => unsafe {
                llvm::LLVMDIBuilderCreateParameterVariable(
                    DIB(self),
                    scope_metadata,
                    name.as_ptr(),
                    name.len(),
                    arg_index as c_uint,
                    file_metadata,
                    loc.line,
                    type_metadata,
                    llvm::Bool::TRUE, // (preserve descriptor during optimizations)
                    DIFlags::FlagZero,
                )
            },
            LocalVariable => unsafe {
                llvm::LLVMDIBuilderCreateAutoVariable(
                    DIB(self),
                    scope_metadata,
                    name.as_ptr(),
                    name.len(),
                    file_metadata,
                    loc.line,
                    type_metadata,
                    llvm::Bool::TRUE, // (preserve descriptor during optimizations)
                    DIFlags::FlagZero,
                    align.bits() as u32,
                )
            },
        }
    }
}
