// See doc.rs for documentation.
mod doc;

use rustc_codegen_ssa::debuginfo::VariableAccess::*;
use rustc_codegen_ssa::debuginfo::VariableKind::*;

use self::utils::{DIB, span_start, create_DIArray, is_node_local_to_unit};
use self::namespace::mangled_name_of_instance;
use self::type_names::compute_debuginfo_type_name;
use self::metadata::{type_metadata, file_metadata, TypeMap};
use self::source_loc::InternalDebugLocation::{self, UnknownLocation};

use crate::llvm;
use crate::llvm::debuginfo::{DIFile, DIType, DIScope, DIBuilder, DISubprogram, DIArray, DIFlags,
    DISPFlags, DILexicalBlock};
use rustc::hir::CodegenFnAttrFlags;
use rustc::hir::def_id::{DefId, CrateNum, LOCAL_CRATE};
use rustc::ty::subst::{SubstsRef, UnpackedKind};

use crate::abi::Abi;
use crate::common::CodegenCx;
use crate::builder::Builder;
use crate::value::Value;
use rustc::ty::{self, ParamEnv, Ty, InstanceDef, Instance};
use rustc::mir;
use rustc::session::config::{self, DebugInfo};
use rustc::util::nodemap::{DefIdMap, FxHashMap, FxHashSet};
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_codegen_ssa::debuginfo::{FunctionDebugContext, MirDebugScope, VariableAccess,
    VariableKind, FunctionDebugContextData, type_names};

use libc::c_uint;
use std::cell::RefCell;
use std::ffi::CString;

use syntax_pos::{self, Span, Pos};
use syntax::ast;
use syntax::symbol::InternedString;
use rustc::ty::layout::{self, LayoutOf, HasTyCtxt};
use rustc_codegen_ssa::traits::*;

pub mod gdb;
mod utils;
mod namespace;
pub mod metadata;
mod create_scope_map;
mod source_loc;

pub use self::create_scope_map::{create_mir_scopes};
pub use self::metadata::create_global_var_metadata;
pub use self::metadata::extend_scope_to_file;
pub use self::source_loc::set_source_location;

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
    created_enum_disr_types: RefCell<FxHashMap<(DefId, layout::Primitive), &'a DIType>>,

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
}

/// Creates any deferred debug metadata nodes
pub fn finalize(cx: &CodegenCx<'_, '_>) {
    if cx.dbg_cx.is_none() {
        return;
    }

    debug!("finalize");

    if gdb::needs_gdb_debug_scripts_section(cx) {
        // Add a .debug_gdb_scripts section to this compile-unit. This will
        // cause GDB to try and load the gdb_load_rust_pretty_printers.py file,
        // which activates the Rust pretty printers for binary this section is
        // contained in.
        gdb::get_or_insert_gdb_debug_scripts_section_global(cx);
    }

    unsafe {
        llvm::LLVMRustDIBuilderFinalize(DIB(cx));
        // Debuginfo generation in LLVM by default uses a higher
        // version of dwarf than macOS currently understands. We can
        // instruct LLVM to emit an older version of dwarf, however,
        // for macOS to understand. For more info see #11352
        // This can be overridden using --llvm-opts -dwarf-version,N.
        // Android has the same issue (#22398)
        if cx.sess().target.target.options.is_like_osx ||
           cx.sess().target.target.options.is_like_android {
            llvm::LLVMRustAddModuleFlag(cx.llmod,
                                        "Dwarf Version\0".as_ptr() as *const _,
                                        2)
        }

        // Indicate that we want CodeView debug information on MSVC
        if cx.sess().target.target.options.is_like_msvc {
            llvm::LLVMRustAddModuleFlag(cx.llmod,
                                        "CodeView\0".as_ptr() as *const _,
                                        1)
        }

        // Prevent bitcode readers from deleting the debug info.
        let ptr = "Debug Info Version\0".as_ptr();
        llvm::LLVMRustAddModuleFlag(cx.llmod, ptr as *const _,
                                    llvm::LLVMRustDebugMetadataVersion());
    };
}

impl DebugInfoBuilderMethods<'tcx> for Builder<'a, 'll, 'tcx> {
    fn declare_local(
        &mut self,
        dbg_context: &FunctionDebugContext<&'ll DISubprogram>,
        variable_name: ast::Name,
        variable_type: Ty<'tcx>,
        scope_metadata: &'ll DIScope,
        variable_access: VariableAccess<'_, &'ll Value>,
        variable_kind: VariableKind,
        span: Span,
    ) {
        assert!(!dbg_context.get_ref(span).source_locations_enabled);
        let cx = self.cx();

        let file = span_start(cx, span).file;
        let file_metadata = file_metadata(cx,
                                          &file.name,
                                          dbg_context.get_ref(span).defining_crate);

        let loc = span_start(cx, span);
        let type_metadata = type_metadata(cx, variable_type, span);

        let (argument_index, dwarf_tag) = match variable_kind {
            ArgumentVariable(index) => (index as c_uint, DW_TAG_arg_variable),
            LocalVariable => (0, DW_TAG_auto_variable)
        };
        let align = cx.align_of(variable_type);

        let name = SmallCStr::new(&variable_name.as_str());
        match (variable_access, &[][..]) {
            (DirectVariable { alloca }, address_operations) |
            (IndirectVariable {alloca, address_operations}, _) => {
                let metadata = unsafe {
                    llvm::LLVMRustDIBuilderCreateVariable(
                        DIB(cx),
                        dwarf_tag,
                        scope_metadata,
                        name.as_ptr(),
                        file_metadata,
                        loc.line as c_uint,
                        type_metadata,
                        cx.sess().opts.optimize != config::OptLevel::No,
                        DIFlags::FlagZero,
                        argument_index,
                        align.bytes() as u32,
                    )
                };
                source_loc::set_debug_location(self,
                    InternalDebugLocation::new(scope_metadata, loc.line, loc.col.to_usize()));
                unsafe {
                    let debug_loc = llvm::LLVMGetCurrentDebugLocation(self.llbuilder);
                    let instr = llvm::LLVMRustDIBuilderInsertDeclareAtEnd(
                        DIB(cx),
                        alloca,
                        metadata,
                        address_operations.as_ptr(),
                        address_operations.len() as c_uint,
                        debug_loc,
                        self.llbb());

                    llvm::LLVMSetInstDebugLocation(self.llbuilder, instr);
                }
                source_loc::set_debug_location(self, UnknownLocation);
            }
        }
    }

    fn set_source_location(
        &mut self,
        debug_context: &mut FunctionDebugContext<&'ll DISubprogram>,
        scope: Option<&'ll DIScope>,
        span: Span,
    ) {
        set_source_location(debug_context, &self, scope, span)
    }
    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {
        gdb::insert_reference_to_gdb_debug_scripts_section_global(self)
    }

    fn set_value_name(&mut self, value: &'ll Value, name: &str) {
        let cname = SmallCStr::new(name);
        unsafe {
            llvm::LLVMSetValueName(value, cname.as_ptr());
        }
    }
}

impl DebugInfoMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        sig: ty::FnSig<'tcx>,
        llfn: &'ll Value,
        mir: &mir::Body<'_>,
    ) -> FunctionDebugContext<&'ll DISubprogram> {
        if self.sess().opts.debuginfo == DebugInfo::None {
            return FunctionDebugContext::DebugInfoDisabled;
        }

        if let InstanceDef::Item(def_id) = instance.def {
            if self.tcx().codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::NO_DEBUG) {
                return FunctionDebugContext::FunctionWithoutDebugInfo;
            }
        }

        let span = mir.span;

        // This can be the case for functions inlined from another crate
        if span.is_dummy() {
            // FIXME(simulacrum): Probably can't happen; remove.
            return FunctionDebugContext::FunctionWithoutDebugInfo;
        }

        let def_id = instance.def_id();
        let containing_scope = get_containing_scope(self, instance);
        let loc = span_start(self, span);
        let file_metadata = file_metadata(self, &loc.file.name, def_id.krate);

        let function_type_metadata = unsafe {
            let fn_signature = get_function_signature(self, sig);
            llvm::LLVMRustDIBuilderCreateSubroutineType(DIB(self), file_metadata, fn_signature)
        };

        // Find the enclosing function, in case this is a closure.
        let def_key = self.tcx().def_key(def_id);
        let mut name = def_key.disambiguated_data.data.to_string();

        let enclosing_fn_def_id = self.tcx().closure_base_def_id(def_id);

        // Get_template_parameters() will append a `<...>` clause to the function
        // name if necessary.
        let generics = self.tcx().generics_of(enclosing_fn_def_id);
        let substs = instance.substs.truncate_to(self.tcx(), generics);
        let template_parameters = get_template_parameters(self,
                                                          &generics,
                                                          substs,
                                                          file_metadata,
                                                          &mut name);

        // Get the linkage_name, which is just the symbol name
        let linkage_name = mangled_name_of_instance(self, instance);

        let scope_line = span_start(self, span).line;

        let function_name = CString::new(name).unwrap();
        let linkage_name = SmallCStr::new(&linkage_name.as_str());

        let mut flags = DIFlags::FlagPrototyped;

        if self.layout_of(sig.output()).abi.is_uninhabited() {
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
            if id == def_id {
                spflags |= DISPFlags::SPFlagMainSubprogram;
            }
        }

        let fn_metadata = unsafe {
            llvm::LLVMRustDIBuilderCreateFunction(
                DIB(self),
                containing_scope,
                function_name.as_ptr(),
                linkage_name.as_ptr(),
                file_metadata,
                loc.line as c_uint,
                function_type_metadata,
                scope_line as c_uint,
                flags,
                spflags,
                llfn,
                template_parameters,
                None)
        };

        // Initialize fn debug context (including scope map and namespace map)
        let fn_debug_context = FunctionDebugContextData {
            fn_metadata,
            source_locations_enabled: false,
            defining_crate: def_id.krate,
        };

        return FunctionDebugContext::RegularContext(fn_debug_context);

        fn get_function_signature<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            sig: ty::FnSig<'tcx>,
        ) -> &'ll DIArray {
            if cx.sess().opts.debuginfo == DebugInfo::Limited {
                return create_DIArray(DIB(cx), &[]);
            }

            let mut signature = Vec::with_capacity(sig.inputs().len() + 1);

            // Return type -- llvm::DIBuilder wants this at index 0
            signature.push(match sig.output().sty {
                ty::Tuple(ref tys) if tys.is_empty() => None,
                _ => Some(type_metadata(cx, sig.output(), syntax_pos::DUMMY_SP))
            });

            let inputs = if sig.abi == Abi::RustCall {
                &sig.inputs()[..sig.inputs().len() - 1]
            } else {
                sig.inputs()
            };

            // Arguments types
            if cx.sess().target.target.options.is_like_msvc {
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
                signature.extend(inputs.iter().map(|&t| {
                    let t = match t.sty {
                        ty::Array(ct, _)
                            if (ct == cx.tcx.types.u8) || cx.layout_of(ct).is_zst() => {
                            cx.tcx.mk_imm_ptr(ct)
                        }
                        _ => t
                    };
                    Some(type_metadata(cx, t, syntax_pos::DUMMY_SP))
                }));
            } else {
                signature.extend(inputs.iter().map(|t| {
                    Some(type_metadata(cx, t, syntax_pos::DUMMY_SP))
                }));
            }

            if sig.abi == Abi::RustCall && !sig.inputs().is_empty() {
                if let ty::Tuple(args) = sig.inputs()[sig.inputs().len() - 1].sty {
                    signature.extend(
                        args.iter().map(|argument_type| {
                            Some(type_metadata(cx, argument_type.expect_ty(), syntax_pos::DUMMY_SP))
                        })
                    );
                }
            }

            create_DIArray(DIB(cx), &signature[..])
        }

        fn get_template_parameters<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            generics: &ty::Generics,
            substs: SubstsRef<'tcx>,
            file_metadata: &'ll DIFile,
            name_to_append_suffix_to: &mut String,
        ) -> &'ll DIArray {
            if substs.types().next().is_none() {
                return create_DIArray(DIB(cx), &[]);
            }

            name_to_append_suffix_to.push('<');
            for (i, actual_type) in substs.types().enumerate() {
                if i != 0 {
                    name_to_append_suffix_to.push_str(",");
                }

                let actual_type =
                    cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), actual_type);
                // Add actual type name to <...> clause of function name
                let actual_type_name = compute_debuginfo_type_name(cx.tcx(),
                                                                   actual_type,
                                                                   true);
                name_to_append_suffix_to.push_str(&actual_type_name[..]);
            }
            name_to_append_suffix_to.push('>');

            // Again, only create type information if full debuginfo is enabled
            let template_params: Vec<_> = if cx.sess().opts.debuginfo == DebugInfo::Full {
                let names = get_parameter_names(cx, generics);
                substs.iter().zip(names).filter_map(|(kind, name)| {
                    if let UnpackedKind::Type(ty) = kind.unpack() {
                        let actual_type =
                            cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), ty);
                        let actual_type_metadata =
                            type_metadata(cx, actual_type, syntax_pos::DUMMY_SP);
                        let name = SmallCStr::new(&name.as_str());
                        Some(unsafe {
                            Some(llvm::LLVMRustDIBuilderCreateTemplateTypeParameter(
                                DIB(cx),
                                None,
                                name.as_ptr(),
                                actual_type_metadata,
                                file_metadata,
                                0,
                                0,
                            ))
                        })
                    } else {
                        None
                    }
                }).collect()
            } else {
                vec![]
            };

            return create_DIArray(DIB(cx), &template_params[..]);
        }

        fn get_parameter_names(cx: &CodegenCx<'_, '_>,
                               generics: &ty::Generics)
                               -> Vec<InternedString> {
            let mut names = generics.parent.map_or(vec![], |def_id| {
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
                        &cx.tcx.type_of(impl_def_id),
                    );

                    // Only "class" methods are generally understood by LLVM,
                    // so avoid methods on other types (e.g., `<*mut T>::null`).
                    match impl_self_ty.sty {
                        ty::Adt(def, ..) if !def.is_box() => {
                            Some(type_metadata(cx, impl_self_ty, syntax_pos::DUMMY_SP))
                        }
                        _ => None
                    }
                } else {
                    // For trait method impls we still use the "parallel namespace"
                    // strategy
                    None
                }
            });

            self_type.unwrap_or_else(|| {
                namespace::item_namespace(cx, DefId {
                    krate: instance.def_id().krate,
                    index: cx.tcx
                             .def_key(instance.def_id())
                             .parent
                             .expect("get_containing_scope: missing parent?")
                })
            })
        }
    }

    fn create_vtable_metadata(
        &self,
        ty: Ty<'tcx>,
        vtable: Self::Value,
    ) {
        metadata::create_vtable_metadata(self, ty, vtable)
    }

    fn create_mir_scopes(
        &self,
        mir: &mir::Body<'_>,
        debug_context: &mut FunctionDebugContext<&'ll DISubprogram>,
    ) -> IndexVec<mir::SourceScope, MirDebugScope<&'ll DIScope>> {
        create_scope_map::create_mir_scopes(self, mir, debug_context)
    }

    fn extend_scope_to_file(
         &self,
         scope_metadata: &'ll DIScope,
         file: &syntax_pos::SourceFile,
         defining_crate: CrateNum,
     ) -> &'ll DILexicalBlock {
         metadata::extend_scope_to_file(&self, scope_metadata, file, defining_crate)
     }

    fn debuginfo_finalize(&self) {
        finalize(self)
    }

    fn debuginfo_upvar_ops_sequence(&self, byte_offset_of_var_in_env: u64) -> [i64; 4] {
        unsafe {
            [llvm::LLVMRustDIBuilderCreateOpDeref(),
             llvm::LLVMRustDIBuilderCreateOpPlusUconst(),
             byte_offset_of_var_in_env as i64,
             llvm::LLVMRustDIBuilderCreateOpDeref()]
        }
    }
}
