#![doc = include_str!("doc.md")]

use rustc_codegen_ssa::mir::debuginfo::VariableKind::*;

use self::metadata::{file_metadata, type_metadata, TypeMap};
use self::metadata::{UNKNOWN_COLUMN_NUMBER, UNKNOWN_LINE_NUMBER};
use self::namespace::mangled_name_of_instance;
use self::utils::{create_DIArray, is_node_local_to_unit, DIB};

use crate::abi::FnAbi;
use crate::builder::Builder;
use crate::common::CodegenCx;
use crate::llvm;
use crate::llvm::debuginfo::{
    DIArray, DIBuilder, DIFile, DIFlags, DILexicalBlock, DILocation, DISPFlags, DIScope, DIType,
    DIVariable,
};
use crate::value::Value;

use rustc_codegen_ssa::debuginfo::type_names;
use rustc_codegen_ssa::mir::debuginfo::{DebugScope, FunctionDebugContext, VariableKind};
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_hir::def_id::{DefId, DefIdMap};
use rustc_index::vec::IndexVec;
use rustc_middle::mir;
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf};
use rustc_middle::ty::subst::{GenericArgKind, SubstsRef};
use rustc_middle::ty::{self, Instance, ParamEnv, Ty, TypeFoldable};
use rustc_session::config::{self, DebugInfo};
use rustc_session::Session;
use rustc_span::symbol::Symbol;
use rustc_span::{self, BytePos, Pos, SourceFile, SourceFileAndLine, Span};
use rustc_target::abi::{Primitive, Size};

use libc::c_uint;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::iter;
use tracing::debug;

mod create_scope_map;
pub mod gdb;
pub mod metadata;
mod namespace;
mod utils;

pub use self::create_scope_map::compute_mir_scopes;
pub use self::metadata::create_global_var_metadata;
pub use self::metadata::extend_scope_to_file;

#[allow(non_upper_case_globals)]
const DW_TAG_auto_variable: c_uint = 0x100;
#[allow(non_upper_case_globals)]
const DW_TAG_arg_variable: c_uint = 0x101;

/// A context object for maintaining all state needed by the debuginfo module.
pub struct CrateDebugContext<'a, 'tcx> {
    llcontext: &'a llvm::Context,
    llmod: &'a llvm::Module,
    builder: &'a mut DIBuilder<'a>,
    created_files: RefCell<FxHashMap<(Option<String>, Option<String>), &'a DIFile>>,
    created_enum_disr_types: RefCell<FxHashMap<(DefId, Primitive), &'a DIType>>,

    type_map: RefCell<TypeMap<'a, 'tcx>>,
    namespace_map: RefCell<DefIdMap<&'a DIScope>>,

    // This collection is used to assert that composite types (structs, enums,
    // ...) have their members only set once:
    composite_types_completed: RefCell<FxHashSet<&'a DIType>>,
}

impl Drop for CrateDebugContext<'a, 'tcx> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustDIBuilderDispose(&mut *(self.builder as *mut _));
        }
    }
}

impl<'a, 'tcx> CrateDebugContext<'a, 'tcx> {
    pub fn new(llmod: &'a llvm::Module) -> Self {
        debug!("CrateDebugContext::new");
        let builder = unsafe { llvm::LLVMRustDIBuilderCreate(llmod) };
        // DIBuilder inherits context from the module, so we'd better use the same one
        let llcontext = unsafe { llvm::LLVMGetModuleContext(llmod) };
        CrateDebugContext {
            llcontext,
            llmod,
            builder,
            created_files: Default::default(),
            created_enum_disr_types: Default::default(),
            type_map: Default::default(),
            namespace_map: RefCell::new(Default::default()),
            composite_types_completed: Default::default(),
        }
    }

    pub fn finalize(&self, sess: &Session) {
        unsafe {
            llvm::LLVMRustDIBuilderFinalize(self.builder);

            // Debuginfo generation in LLVM by default uses a higher
            // version of dwarf than macOS currently understands. We can
            // instruct LLVM to emit an older version of dwarf, however,
            // for macOS to understand. For more info see #11352
            // This can be overridden using --llvm-opts -dwarf-version,N.
            // Android has the same issue (#22398)
            if let Some(version) = sess.target.dwarf_version {
                llvm::LLVMRustAddModuleFlag(self.llmod, "Dwarf Version\0".as_ptr().cast(), version)
            }

            // Indicate that we want CodeView debug information on MSVC
            if sess.target.is_like_msvc {
                llvm::LLVMRustAddModuleFlag(self.llmod, "CodeView\0".as_ptr().cast(), 1)
            }

            // Prevent bitcode readers from deleting the debug info.
            let ptr = "Debug Info Version\0".as_ptr();
            llvm::LLVMRustAddModuleFlag(
                self.llmod,
                ptr.cast(),
                llvm::LLVMRustDebugMetadataVersion(),
            );
        }
    }
}

/// Creates any deferred debug metadata nodes
pub fn finalize(cx: &CodegenCx<'_, '_>) {
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

impl DebugInfoBuilderMethods for Builder<'a, 'll, 'tcx> {
    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `dbg`, `debug`, `debuginfo`, `debug_info` etc.).
    fn dbg_var_addr(
        &mut self,
        dbg_var: &'ll DIVariable,
        dbg_loc: &'ll DILocation,
        variable_alloca: Self::Value,
        direct_offset: Size,
        indirect_offsets: &[Size],
    ) {
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

        unsafe {
            // FIXME(eddyb) replace `llvm.dbg.declare` with `llvm.dbg.addr`.
            llvm::LLVMRustDIBuilderInsertDeclareAtEnd(
                DIB(self.cx()),
                variable_alloca,
                dbg_var,
                addr_ops.as_ptr(),
                addr_ops.len() as c_uint,
                dbg_loc,
                self.llbb(),
            );
        }
    }

    fn set_dbg_loc(&mut self, dbg_loc: &'ll DILocation) {
        unsafe {
            let dbg_loc_as_llval = llvm::LLVMRustMetadataAsValue(self.cx().llcx, dbg_loc);
            llvm::LLVMSetCurrentDebugLocation(self.llbuilder, dbg_loc_as_llval);
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
}

/// A source code location used to generate debug information.
// FIXME(eddyb) rename this to better indicate it's a duplicate of
// `rustc_span::Loc` rather than `DILocation`, perhaps by making
// `lookup_char_pos` return the right information instead.
pub struct DebugLoc {
    /// Information about the original source file.
    pub file: Lrc<SourceFile>,
    /// The (1-based) line number.
    pub line: u32,
    /// The (1-based) column number.
    pub col: u32,
}

impl CodegenCx<'ll, '_> {
    /// Looks up debug source information about a `BytePos`.
    // FIXME(eddyb) rename this to better indicate it's a duplicate of
    // `lookup_char_pos` rather than `dbg_loc`, perhaps by making
    // `lookup_char_pos` return the right information instead.
    pub fn lookup_debug_loc(&self, pos: BytePos) -> DebugLoc {
        let (file, line, col) = match self.sess().source_map().lookup_line(pos) {
            Ok(SourceFileAndLine { sf: file, line }) => {
                let line_pos = file.line_begin_pos(pos);

                // Use 1-based indexing.
                let line = (line + 1) as u32;
                let col = (pos - line_pos).to_u32() + 1;

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
}

impl DebugInfoMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        llfn: &'ll Value,
        mir: &mir::Body<'tcx>,
    ) -> Option<FunctionDebugContext<&'ll DIScope, &'ll DILocation>> {
        if self.sess().opts.debuginfo == DebugInfo::None {
            return None;
        }

        // Initialize fn debug context (including scopes).
        // FIXME(eddyb) figure out a way to not need `Option` for `dbg_scope`.
        let empty_scope = DebugScope {
            dbg_scope: None,
            inlined_at: None,
            file_start_pos: BytePos(0),
            file_end_pos: BytePos(0),
        };
        let mut fn_debug_context =
            FunctionDebugContext { scopes: IndexVec::from_elem(empty_scope, &mir.source_scopes) };

        // Fill in all the scopes, with the information from the MIR body.
        compute_mir_scopes(
            self,
            instance,
            mir,
            self.dbg_scope_fn(instance, fn_abi, Some(llfn)),
            &mut fn_debug_context,
        );

        Some(fn_debug_context)
    }

    fn dbg_scope_fn(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        maybe_definition_llfn: Option<&'ll Value>,
    ) -> &'ll DIScope {
        let def_id = instance.def_id();
        let containing_scope = get_containing_scope(self, instance);
        let span = self.tcx.def_span(def_id);
        let loc = self.lookup_debug_loc(span.lo());
        let file_metadata = file_metadata(self, &loc.file);

        let function_type_metadata = unsafe {
            let fn_signature = get_function_signature(self, fn_abi);
            llvm::LLVMRustDIBuilderCreateSubroutineType(DIB(self), fn_signature)
        };

        let mut name = String::new();
        type_names::push_item_name(self.tcx(), def_id, false, &mut name);

        // Find the enclosing function, in case this is a closure.
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
        if let Some((id, _)) = self.tcx.entry_fn(()) {
            if id == def_id {
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
                loc.line,
                function_type_metadata,
                scope_line,
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
            type_names::push_generic_params(
                cx.tcx,
                cx.tcx.normalize_erasing_regions(ty::ParamEnv::reveal_all(), substs),
                name_to_append_suffix_to,
            );

            if substs.types().next().is_none() {
                return create_DIArray(DIB(cx), &[]);
            }

            // Again, only create type information if full debuginfo is enabled
            let template_params: Vec<_> = if cx.sess().opts.debuginfo == DebugInfo::Full {
                let names = get_parameter_names(cx, generics);
                iter::zip(substs, names)
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
            let mut names = generics.parent.map_or_else(Vec::new, |def_id| {
                get_parameter_names(cx, cx.tcx.generics_of(def_id))
            });
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
                        cx.tcx.type_of(impl_def_id),
                    );

                    // Only "class" methods are generally understood by LLVM,
                    // so avoid methods on other types (e.g., `<*mut T>::null`).
                    match impl_self_ty.kind() {
                        ty::Adt(def, ..) if !def.is_box() => {
                            // Again, only create type information if full debuginfo is enabled
                            if cx.sess().opts.debuginfo == DebugInfo::Full
                                && !impl_self_ty.definitely_needs_subst(cx.tcx)
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
        }
    }

    fn dbg_loc(
        &self,
        scope: &'ll DIScope,
        inlined_at: Option<&'ll DILocation>,
        span: Span,
    ) -> &'ll DILocation {
        let DebugLoc { line, col, .. } = self.lookup_debug_loc(span.lo());

        unsafe { llvm::LLVMRustDIBuilderCreateDebugLocation(line, col, scope, inlined_at) }
    }

    fn create_vtable_metadata(&self, ty: Ty<'tcx>, vtable: Self::Value) {
        metadata::create_vtable_metadata(self, ty, vtable)
    }

    fn extend_scope_to_file(
        &self,
        scope_metadata: &'ll DIScope,
        file: &rustc_span::SourceFile,
    ) -> &'ll DILexicalBlock {
        metadata::extend_scope_to_file(&self, scope_metadata, file)
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

        let type_metadata = type_metadata(self, variable_type, span);

        let (argument_index, dwarf_tag) = match variable_kind {
            ArgumentVariable(index) => (index as c_uint, DW_TAG_arg_variable),
            LocalVariable => (0, DW_TAG_auto_variable),
        };
        let align = self.align_of(variable_type);

        let name = variable_name.as_str();
        unsafe {
            llvm::LLVMRustDIBuilderCreateVariable(
                DIB(self),
                dwarf_tag,
                scope_metadata,
                name.as_ptr().cast(),
                name.len(),
                file_metadata,
                loc.line,
                type_metadata,
                true,
                DIFlags::FlagZero,
                argument_index,
                align.bytes() as u32,
            )
        }
    }
}
