// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// See doc.rs for documentation.
mod doc;

use self::VariableAccess::*;
use self::VariableKind::*;

use self::utils::{DIB, span_start, create_DIArray, is_node_local_to_unit};
use self::namespace::mangled_name_of_item;
use self::type_names::compute_debuginfo_type_name;
use self::metadata::{type_metadata, file_metadata, TypeMap};
use self::source_loc::InternalDebugLocation::{self, UnknownLocation};

use llvm;
use llvm::{ModuleRef, ContextRef, ValueRef};
use llvm::debuginfo::{DIFile, DIType, DIScope, DIBuilderRef, DISubprogram, DIArray, DIFlags};
use rustc::hir::def_id::DefId;
use rustc::ty::subst::Substs;

use abi::Abi;
use common::CrateContext;
use builder::Builder;
use monomorphize::{self, Instance};
use rustc::ty::{self, Ty};
use rustc::mir;
use session::config::{self, FullDebugInfo, LimitedDebugInfo, NoDebugInfo};
use util::nodemap::{DefIdMap, FxHashMap, FxHashSet};

use libc::c_uint;
use std::cell::{Cell, RefCell};
use std::ffi::CString;
use std::ptr;

use syntax_pos::{self, Span, Pos};
use syntax::ast;
use rustc::ty::layout;

pub mod gdb;
mod utils;
mod namespace;
mod type_names;
pub mod metadata;
mod create_scope_map;
mod source_loc;

pub use self::create_scope_map::{create_mir_scopes, MirDebugScope};
pub use self::source_loc::start_emitting_source_locations;
pub use self::metadata::create_global_var_metadata;
pub use self::metadata::extend_scope_to_file;
pub use self::source_loc::set_source_location;

#[allow(non_upper_case_globals)]
const DW_TAG_auto_variable: c_uint = 0x100;
#[allow(non_upper_case_globals)]
const DW_TAG_arg_variable: c_uint = 0x101;

/// A context object for maintaining all state needed by the debuginfo module.
pub struct CrateDebugContext<'tcx> {
    llcontext: ContextRef,
    builder: DIBuilderRef,
    created_files: RefCell<FxHashMap<String, DIFile>>,
    created_enum_disr_types: RefCell<FxHashMap<(DefId, layout::Integer), DIType>>,

    type_map: RefCell<TypeMap<'tcx>>,
    namespace_map: RefCell<DefIdMap<DIScope>>,

    // This collection is used to assert that composite types (structs, enums,
    // ...) have their members only set once:
    composite_types_completed: RefCell<FxHashSet<DIType>>,
}

impl<'tcx> CrateDebugContext<'tcx> {
    pub fn new(llmod: ModuleRef) -> CrateDebugContext<'tcx> {
        debug!("CrateDebugContext::new");
        let builder = unsafe { llvm::LLVMRustDIBuilderCreate(llmod) };
        // DIBuilder inherits context from the module, so we'd better use the same one
        let llcontext = unsafe { llvm::LLVMGetModuleContext(llmod) };
        CrateDebugContext {
            llcontext: llcontext,
            builder: builder,
            created_files: RefCell::new(FxHashMap()),
            created_enum_disr_types: RefCell::new(FxHashMap()),
            type_map: RefCell::new(TypeMap::new()),
            namespace_map: RefCell::new(DefIdMap()),
            composite_types_completed: RefCell::new(FxHashSet()),
        }
    }
}

pub enum FunctionDebugContext {
    RegularContext(FunctionDebugContextData),
    DebugInfoDisabled,
    FunctionWithoutDebugInfo,
}

impl FunctionDebugContext {
    fn get_ref<'a>(&'a self, span: Span) -> &'a FunctionDebugContextData {
        match *self {
            FunctionDebugContext::RegularContext(ref data) => data,
            FunctionDebugContext::DebugInfoDisabled => {
                span_bug!(span, "{}", FunctionDebugContext::debuginfo_disabled_message());
            }
            FunctionDebugContext::FunctionWithoutDebugInfo => {
                span_bug!(span, "{}", FunctionDebugContext::should_be_ignored_message());
            }
        }
    }

    fn debuginfo_disabled_message() -> &'static str {
        "debuginfo: Error trying to access FunctionDebugContext although debug info is disabled!"
    }

    fn should_be_ignored_message() -> &'static str {
        "debuginfo: Error trying to access FunctionDebugContext for function that should be \
         ignored by debug info!"
    }
}

pub struct FunctionDebugContextData {
    fn_metadata: DISubprogram,
    source_locations_enabled: Cell<bool>,
}

pub enum VariableAccess<'a> {
    // The llptr given is an alloca containing the variable's value
    DirectVariable { alloca: ValueRef },
    // The llptr given is an alloca containing the start of some pointer chain
    // leading to the variable's content.
    IndirectVariable { alloca: ValueRef, address_operations: &'a [i64] }
}

pub enum VariableKind {
    ArgumentVariable(usize /*index*/),
    LocalVariable,
    CapturedVariable,
}

/// Create any deferred debug metadata nodes
pub fn finalize(cx: &CrateContext) {
    if cx.dbg_cx().is_none() {
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
        llvm::LLVMRustDIBuilderDispose(DIB(cx));
        // Debuginfo generation in LLVM by default uses a higher
        // version of dwarf than OS X currently understands. We can
        // instruct LLVM to emit an older version of dwarf, however,
        // for OS X to understand. For more info see #11352
        // This can be overridden using --llvm-opts -dwarf-version,N.
        // Android has the same issue (#22398)
        if cx.sess().target.target.options.is_like_osx ||
           cx.sess().target.target.options.is_like_android {
            llvm::LLVMRustAddModuleFlag(cx.llmod(),
                                        "Dwarf Version\0".as_ptr() as *const _,
                                        2)
        }

        // Indicate that we want CodeView debug information on MSVC
        if cx.sess().target.target.options.is_like_msvc {
            llvm::LLVMRustAddModuleFlag(cx.llmod(),
                                        "CodeView\0".as_ptr() as *const _,
                                        1)
        }

        // Prevent bitcode readers from deleting the debug info.
        let ptr = "Debug Info Version\0".as_ptr();
        llvm::LLVMRustAddModuleFlag(cx.llmod(), ptr as *const _,
                                    llvm::LLVMRustDebugMetadataVersion());
    };
}

/// Creates the function-specific debug context.
///
/// Returns the FunctionDebugContext for the function which holds state needed
/// for debug info creation. The function may also return another variant of the
/// FunctionDebugContext enum which indicates why no debuginfo should be created
/// for the function.
pub fn create_function_debug_context<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                               instance: Instance<'tcx>,
                                               sig: &ty::FnSig<'tcx>,
                                               abi: Abi,
                                               llfn: ValueRef,
                                               mir: &mir::Mir) -> FunctionDebugContext {
    if cx.sess().opts.debuginfo == NoDebugInfo {
        return FunctionDebugContext::DebugInfoDisabled;
    }

    for attr in cx.tcx().get_attrs(instance.def).iter() {
        if attr.check_name("no_debug") {
            return FunctionDebugContext::FunctionWithoutDebugInfo;
        }
    }

    let containing_scope = get_containing_scope(cx, instance);
    let span = mir.span;

    // This can be the case for functions inlined from another crate
    if span == syntax_pos::DUMMY_SP {
        // FIXME(simulacrum): Probably can't happen; remove.
        return FunctionDebugContext::FunctionWithoutDebugInfo;
    }

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, &loc.file.name, &loc.file.abs_path);

    let function_type_metadata = unsafe {
        let fn_signature = get_function_signature(cx, sig, abi);
        llvm::LLVMRustDIBuilderCreateSubroutineType(DIB(cx), file_metadata, fn_signature)
    };

    // Find the enclosing function, in case this is a closure.
    let def_key = cx.tcx().def_key(instance.def);
    let mut name = def_key.disambiguated_data.data.to_string();
    let name_len = name.len();

    let fn_def_id = cx.tcx().closure_base_def_id(instance.def);

    // Get_template_parameters() will append a `<...>` clause to the function
    // name if necessary.
    let generics = cx.tcx().item_generics(fn_def_id);
    let substs = instance.substs.truncate_to(cx.tcx(), generics);
    let template_parameters = get_template_parameters(cx,
                                                      &generics,
                                                      substs,
                                                      file_metadata,
                                                      &mut name);

    // Build the linkage_name out of the item path and "template" parameters.
    let linkage_name = mangled_name_of_item(cx, instance.def, &name[name_len..]);

    let scope_line = span_start(cx, span).line;

    let local_id = cx.tcx().map.as_local_node_id(instance.def);
    let is_local_to_unit = local_id.map_or(false, |id| is_node_local_to_unit(cx, id));

    let function_name = CString::new(name).unwrap();
    let linkage_name = CString::new(linkage_name).unwrap();

    let fn_metadata = unsafe {
        llvm::LLVMRustDIBuilderCreateFunction(
            DIB(cx),
            containing_scope,
            function_name.as_ptr(),
            linkage_name.as_ptr(),
            file_metadata,
            loc.line as c_uint,
            function_type_metadata,
            is_local_to_unit,
            true,
            scope_line as c_uint,
            DIFlags::FlagPrototyped,
            cx.sess().opts.optimize != config::OptLevel::No,
            llfn,
            template_parameters,
            ptr::null_mut())
    };

    // Initialize fn debug context (including scope map and namespace map)
    let fn_debug_context = FunctionDebugContextData {
        fn_metadata: fn_metadata,
        source_locations_enabled: Cell::new(false),
    };

    return FunctionDebugContext::RegularContext(fn_debug_context);

    fn get_function_signature<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                        sig: &ty::FnSig<'tcx>,
                                        abi: Abi) -> DIArray {
        if cx.sess().opts.debuginfo == LimitedDebugInfo {
            return create_DIArray(DIB(cx), &[]);
        }

        let mut signature = Vec::with_capacity(sig.inputs().len() + 1);

        // Return type -- llvm::DIBuilder wants this at index 0
        signature.push(match sig.output().sty {
            ty::TyTuple(ref tys) if tys.is_empty() => ptr::null_mut(),
            _ => type_metadata(cx, sig.output(), syntax_pos::DUMMY_SP)
        });

        let inputs = if abi == Abi::RustCall {
            &sig.inputs()[..sig.inputs().len() - 1]
        } else {
            sig.inputs()
        };

        // Arguments types
        for &argument_type in inputs {
            signature.push(type_metadata(cx, argument_type, syntax_pos::DUMMY_SP));
        }

        if abi == Abi::RustCall && !sig.inputs().is_empty() {
            if let ty::TyTuple(args) = sig.inputs()[sig.inputs().len() - 1].sty {
                for &argument_type in args {
                    signature.push(type_metadata(cx, argument_type, syntax_pos::DUMMY_SP));
                }
            }
        }

        return create_DIArray(DIB(cx), &signature[..]);
    }

    fn get_template_parameters<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                         generics: &ty::Generics<'tcx>,
                                         substs: &Substs<'tcx>,
                                         file_metadata: DIFile,
                                         name_to_append_suffix_to: &mut String)
                                         -> DIArray
    {
        if substs.types().next().is_none() {
            return create_DIArray(DIB(cx), &[]);
        }

        name_to_append_suffix_to.push('<');
        for (i, actual_type) in substs.types().enumerate() {
            if i != 0 {
                name_to_append_suffix_to.push_str(",");
            }

            let actual_type = cx.tcx().normalize_associated_type(&actual_type);
            // Add actual type name to <...> clause of function name
            let actual_type_name = compute_debuginfo_type_name(cx,
                                                               actual_type,
                                                               true);
            name_to_append_suffix_to.push_str(&actual_type_name[..]);
        }
        name_to_append_suffix_to.push('>');

        // Again, only create type information if full debuginfo is enabled
        let template_params: Vec<_> = if cx.sess().opts.debuginfo == FullDebugInfo {
            let names = get_type_parameter_names(cx, generics);
            substs.types().zip(names).map(|(ty, name)| {
                let actual_type = cx.tcx().normalize_associated_type(&ty);
                let actual_type_metadata = type_metadata(cx, actual_type, syntax_pos::DUMMY_SP);
                let name = CString::new(name.as_str().as_bytes()).unwrap();
                unsafe {
                    llvm::LLVMRustDIBuilderCreateTemplateTypeParameter(
                        DIB(cx),
                        ptr::null_mut(),
                        name.as_ptr(),
                        actual_type_metadata,
                        file_metadata,
                        0,
                        0)
                }
            }).collect()
        } else {
            vec![]
        };

        return create_DIArray(DIB(cx), &template_params[..]);
    }

    fn get_type_parameter_names<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                          generics: &ty::Generics<'tcx>)
                                          -> Vec<ast::Name> {
        let mut names = generics.parent.map_or(vec![], |def_id| {
            get_type_parameter_names(cx, cx.tcx().item_generics(def_id))
        });
        names.extend(generics.types.iter().map(|param| param.name));
        names
    }

    fn get_containing_scope<'ccx, 'tcx>(cx: &CrateContext<'ccx, 'tcx>,
                                        instance: Instance<'tcx>)
                                        -> DIScope {
        // First, let's see if this is a method within an inherent impl. Because
        // if yes, we want to make the result subroutine DIE a child of the
        // subroutine's self-type.
        let self_type = cx.tcx().impl_of_method(instance.def).and_then(|impl_def_id| {
            // If the method does *not* belong to a trait, proceed
            if cx.tcx().trait_id_of_impl(impl_def_id).is_none() {
                let impl_self_ty = cx.tcx().item_type(impl_def_id);
                let impl_self_ty = cx.tcx().erase_regions(&impl_self_ty);
                let impl_self_ty = monomorphize::apply_param_substs(cx.shared(),
                                                                    instance.substs,
                                                                    &impl_self_ty);

                // Only "class" methods are generally understood by LLVM,
                // so avoid methods on other types (e.g. `<*mut T>::null`).
                match impl_self_ty.sty {
                    ty::TyAdt(..) => {
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
                krate: instance.def.krate,
                index: cx.tcx()
                         .def_key(instance.def)
                         .parent
                         .expect("get_containing_scope: missing parent?")
            })
        })
    }
}

pub fn declare_local<'a, 'tcx>(bcx: &Builder<'a, 'tcx>,
                               dbg_context: &FunctionDebugContext,
                               variable_name: ast::Name,
                               variable_type: Ty<'tcx>,
                               scope_metadata: DIScope,
                               variable_access: VariableAccess,
                               variable_kind: VariableKind,
                               span: Span) {
    let cx = bcx.ccx;

    let file = span_start(cx, span).file;
    let filename = file.name.clone();
    let file_metadata = file_metadata(cx, &filename[..], &file.abs_path);

    let loc = span_start(cx, span);
    let type_metadata = type_metadata(cx, variable_type, span);

    let (argument_index, dwarf_tag) = match variable_kind {
        ArgumentVariable(index) => (index as c_uint, DW_TAG_arg_variable),
        LocalVariable    |
        CapturedVariable => (0, DW_TAG_auto_variable)
    };
    let align = ::type_of::align_of(cx, variable_type);

    let name = CString::new(variable_name.as_str().as_bytes()).unwrap();
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
                    align as u64,
                )
            };
            source_loc::set_debug_location(bcx,
                InternalDebugLocation::new(scope_metadata, loc.line, loc.col.to_usize()));
            unsafe {
                let debug_loc = llvm::LLVMGetCurrentDebugLocation(bcx.llbuilder);
                let instr = llvm::LLVMRustDIBuilderInsertDeclareAtEnd(
                    DIB(cx),
                    alloca,
                    metadata,
                    address_operations.as_ptr(),
                    address_operations.len() as c_uint,
                    debug_loc,
                    bcx.llbb());

                llvm::LLVMSetInstDebugLocation(bcx.llbuilder, instr);
            }
        }
    }

    match variable_kind {
        ArgumentVariable(_) | CapturedVariable => {
            assert!(!dbg_context.get_ref(span).source_locations_enabled.get());
            source_loc::set_debug_location(bcx, UnknownLocation);
        }
        _ => { /* nothing to do */ }
    }
}
