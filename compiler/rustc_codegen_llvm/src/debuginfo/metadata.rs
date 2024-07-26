use self::type_map::DINodeCreationResult;
use self::type_map::Stub;
use self::type_map::UniqueTypeId;

use super::namespace::mangled_name_of_instance;
use super::type_names::{compute_debuginfo_type_name, compute_debuginfo_vtable_name};
use super::utils::{
    create_DIArray, debug_context, get_namespace_for_item, is_node_local_to_unit, DIB,
};
use super::CodegenUnitDebugContext;

use crate::abi;
use crate::common::CodegenCx;
use crate::debuginfo::metadata::type_map::build_type_with_children;
use crate::debuginfo::utils::fat_pointer_kind;
use crate::debuginfo::utils::FatPtrKind;
use crate::llvm;
use crate::llvm::debuginfo::{
    DIDescriptor, DIFile, DIFlags, DILexicalBlock, DIScope, DIType, DebugEmissionKind,
    DebugNameTableKind,
};
use crate::value::Value;

use rustc_codegen_ssa::debuginfo::type_names::cpp_like_debuginfo;
use rustc_codegen_ssa::debuginfo::type_names::VTableNameKind;
use rustc_codegen_ssa::traits::*;
use rustc_fs_util::path_to_c_string;
use rustc_hir::def::CtorKind;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::{
    self, AdtKind, CoroutineArgsExt, Instance, ParamEnv, PolyExistentialTraitRef, Ty, TyCtxt,
    Visibility,
};
use rustc_session::config::{self, DebugInfo, Lto};
use rustc_span::symbol::Symbol;
use rustc_span::{hygiene, FileName, DUMMY_SP};
use rustc_span::{FileNameDisplayPreference, SourceFile};
use rustc_symbol_mangling::typeid_for_trait_ref;
use rustc_target::abi::{Align, Size};
use rustc_target::spec::DebuginfoKind;
use smallvec::smallvec;
use tracing::{debug, instrument};

use libc::{c_char, c_longlong, c_uint};
use std::borrow::Cow;
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
use std::iter;
use std::path::{Path, PathBuf};
use std::ptr;

impl PartialEq for llvm::Metadata {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for llvm::Metadata {}

impl Hash for llvm::Metadata {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        (self as *const Self).hash(hasher);
    }
}

impl fmt::Debug for llvm::Metadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self as *const Self).fmt(f)
    }
}

// From DWARF 5.
// See http://www.dwarfstd.org/ShowIssue.php?issue=140129.1.
const DW_LANG_RUST: c_uint = 0x1c;
#[allow(non_upper_case_globals)]
const DW_ATE_boolean: c_uint = 0x02;
#[allow(non_upper_case_globals)]
const DW_ATE_float: c_uint = 0x04;
#[allow(non_upper_case_globals)]
const DW_ATE_signed: c_uint = 0x05;
#[allow(non_upper_case_globals)]
const DW_ATE_unsigned: c_uint = 0x07;
#[allow(non_upper_case_globals)]
const DW_ATE_UTF: c_uint = 0x10;

pub(super) const UNKNOWN_LINE_NUMBER: c_uint = 0;
pub(super) const UNKNOWN_COLUMN_NUMBER: c_uint = 0;

const NO_SCOPE_METADATA: Option<&DIScope> = None;
/// A function that returns an empty list of generic parameter debuginfo nodes.
const NO_GENERICS: for<'ll> fn(&CodegenCx<'ll, '_>) -> SmallVec<&'ll DIType> = |_| SmallVec::new();

// SmallVec is used quite a bit in this module, so create a shorthand.
// The actual number of elements is not so important.
pub type SmallVec<T> = smallvec::SmallVec<[T; 16]>;

mod enums;
mod type_map;

pub(crate) use type_map::TypeMap;

/// Returns from the enclosing function if the type debuginfo node with the given
/// unique ID can be found in the type map.
macro_rules! return_if_di_node_created_in_meantime {
    ($cx: expr, $unique_type_id: expr) => {
        if let Some(di_node) = debug_context($cx).type_map.di_node_for_unique_id($unique_type_id) {
            return DINodeCreationResult::new(di_node, true);
        }
    };
}

/// Extract size and alignment from a TyAndLayout.
#[inline]
fn size_and_align_of(ty_and_layout: TyAndLayout<'_>) -> (Size, Align) {
    (ty_and_layout.size, ty_and_layout.align.abi)
}

/// Creates debuginfo for a fixed size array (e.g. `[u64; 123]`).
/// For slices (that is, "arrays" of unknown size) use [build_slice_type_di_node].
fn build_fixed_size_array_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
    array_type: Ty<'tcx>,
) -> DINodeCreationResult<'ll> {
    let ty::Array(element_type, len) = array_type.kind() else {
        bug!("build_fixed_size_array_di_node() called with non-ty::Array type `{:?}`", array_type)
    };

    let element_type_di_node = type_di_node(cx, *element_type);

    return_if_di_node_created_in_meantime!(cx, unique_type_id);

    let (size, align) = cx.size_and_align_of(array_type);

    let upper_bound = len.eval_target_usize(cx.tcx, ty::ParamEnv::reveal_all()) as c_longlong;

    let subrange =
        unsafe { Some(llvm::LLVMRustDIBuilderGetOrCreateSubrange(DIB(cx), 0, upper_bound)) };

    let subscripts = create_DIArray(DIB(cx), &[subrange]);
    let di_node = unsafe {
        llvm::LLVMRustDIBuilderCreateArrayType(
            DIB(cx),
            size.bits(),
            align.bits() as u32,
            element_type_di_node,
            subscripts,
        )
    };

    DINodeCreationResult::new(di_node, false)
}

/// Creates debuginfo for built-in pointer-like things:
///
///  - ty::Ref
///  - ty::RawPtr
///  - ty::Adt in the case it's Box
///
/// At some point we might want to remove the special handling of Box
/// and treat it the same as other smart pointers (like Rc, Arc, ...).
fn build_pointer_or_reference_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    ptr_type: Ty<'tcx>,
    pointee_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    // The debuginfo generated by this function is only valid if `ptr_type` is really just
    // a (fat) pointer. Make sure it is not called for e.g. `Box<T, NonZSTAllocator>`.
    assert_eq!(
        cx.size_and_align_of(ptr_type),
        cx.size_and_align_of(Ty::new_mut_ptr(cx.tcx, pointee_type))
    );

    let pointee_type_di_node = type_di_node(cx, pointee_type);

    return_if_di_node_created_in_meantime!(cx, unique_type_id);

    let data_layout = &cx.tcx.data_layout;
    let ptr_type_debuginfo_name = compute_debuginfo_type_name(cx.tcx, ptr_type, true);

    match fat_pointer_kind(cx, pointee_type) {
        None => {
            // This is a thin pointer. Create a regular pointer type and give it the correct name.
            assert_eq!(
                (data_layout.pointer_size, data_layout.pointer_align.abi),
                cx.size_and_align_of(ptr_type),
                "ptr_type={ptr_type}, pointee_type={pointee_type}",
            );

            let di_node = unsafe {
                llvm::LLVMRustDIBuilderCreatePointerType(
                    DIB(cx),
                    pointee_type_di_node,
                    data_layout.pointer_size.bits(),
                    data_layout.pointer_align.abi.bits() as u32,
                    0, // Ignore DWARF address space.
                    ptr_type_debuginfo_name.as_ptr().cast(),
                    ptr_type_debuginfo_name.len(),
                )
            };

            DINodeCreationResult { di_node, already_stored_in_typemap: false }
        }
        Some(fat_pointer_kind) => {
            type_map::build_type_with_children(
                cx,
                type_map::stub(
                    cx,
                    Stub::Struct,
                    unique_type_id,
                    &ptr_type_debuginfo_name,
                    cx.size_and_align_of(ptr_type),
                    NO_SCOPE_METADATA,
                    DIFlags::FlagZero,
                ),
                |cx, owner| {
                    // FIXME: If this fat pointer is a `Box` then we don't want to use its
                    //        type layout and instead use the layout of the raw pointer inside
                    //        of it.
                    //        The proper way to handle this is to not treat Box as a pointer
                    //        at all and instead emit regular struct debuginfo for it. We just
                    //        need to make sure that we don't break existing debuginfo consumers
                    //        by doing that (at least not without a warning period).
                    let layout_type = if ptr_type.is_box() {
                        Ty::new_mut_ptr(cx.tcx, pointee_type)
                    } else {
                        ptr_type
                    };

                    let layout = cx.layout_of(layout_type);
                    let addr_field = layout.field(cx, abi::FAT_PTR_ADDR);
                    let extra_field = layout.field(cx, abi::FAT_PTR_EXTRA);

                    let (addr_field_name, extra_field_name) = match fat_pointer_kind {
                        FatPtrKind::Dyn => ("pointer", "vtable"),
                        FatPtrKind::Slice => ("data_ptr", "length"),
                    };

                    assert_eq!(abi::FAT_PTR_ADDR, 0);
                    assert_eq!(abi::FAT_PTR_EXTRA, 1);

                    // The data pointer type is a regular, thin pointer, regardless of whether this
                    // is a slice or a trait object.
                    let data_ptr_type_di_node = unsafe {
                        llvm::LLVMRustDIBuilderCreatePointerType(
                            DIB(cx),
                            pointee_type_di_node,
                            addr_field.size.bits(),
                            addr_field.align.abi.bits() as u32,
                            0, // Ignore DWARF address space.
                            std::ptr::null(),
                            0,
                        )
                    };

                    smallvec![
                        build_field_di_node(
                            cx,
                            owner,
                            addr_field_name,
                            (addr_field.size, addr_field.align.abi),
                            layout.fields.offset(abi::FAT_PTR_ADDR),
                            DIFlags::FlagZero,
                            data_ptr_type_di_node,
                        ),
                        build_field_di_node(
                            cx,
                            owner,
                            extra_field_name,
                            (extra_field.size, extra_field.align.abi),
                            layout.fields.offset(abi::FAT_PTR_EXTRA),
                            DIFlags::FlagZero,
                            type_di_node(cx, extra_field.ty),
                        ),
                    ]
                },
                NO_GENERICS,
            )
        }
    }
}

fn build_subroutine_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    // It's possible to create a self-referential
    // type in Rust by using 'impl trait':
    //
    // fn foo() -> impl Copy { foo }
    //
    // Unfortunately LLVM's API does not allow us to create recursive subroutine types.
    // In order to work around that restriction we place a marker type in the type map,
    // before creating the actual type. If the actual type is recursive, it will hit the
    // marker type. So we end up with a type that looks like
    //
    // fn foo() -> <recursive_type>
    //
    // Once that is created, we replace the marker in the typemap with the actual type.
    debug_context(cx)
        .type_map
        .unique_id_to_di_node
        .borrow_mut()
        .insert(unique_type_id, recursion_marker_type_di_node(cx));

    let fn_ty = unique_type_id.expect_ty();
    let signature = cx
        .tcx
        .normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), fn_ty.fn_sig(cx.tcx));

    let signature_di_nodes: SmallVec<_> = iter::once(
        // return type
        match signature.output().kind() {
            ty::Tuple(tys) if tys.is_empty() => {
                // this is a "void" function
                None
            }
            _ => Some(type_di_node(cx, signature.output())),
        },
    )
    .chain(
        // regular arguments
        signature.inputs().iter().map(|&argument_type| Some(type_di_node(cx, argument_type))),
    )
    .collect();

    debug_context(cx).type_map.unique_id_to_di_node.borrow_mut().remove(&unique_type_id);

    let fn_di_node = unsafe {
        llvm::LLVMRustDIBuilderCreateSubroutineType(
            DIB(cx),
            create_DIArray(DIB(cx), &signature_di_nodes[..]),
        )
    };

    // This is actually a function pointer, so wrap it in pointer DI.
    let name = compute_debuginfo_type_name(cx.tcx, fn_ty, false);
    let (size, align) = match fn_ty.kind() {
        ty::FnDef(..) => (0, 1),
        ty::FnPtr(..) => (
            cx.tcx.data_layout.pointer_size.bits(),
            cx.tcx.data_layout.pointer_align.abi.bits() as u32,
        ),
        _ => unreachable!(),
    };
    let di_node = unsafe {
        llvm::LLVMRustDIBuilderCreatePointerType(
            DIB(cx),
            fn_di_node,
            size,
            align,
            0, // Ignore DWARF address space.
            name.as_ptr().cast(),
            name.len(),
        )
    };

    DINodeCreationResult::new(di_node, false)
}

/// Create debuginfo for `dyn SomeTrait` types. Currently these are empty structs
/// we with the correct type name (e.g. "dyn SomeTrait<Foo, Item=u32> + Sync").
fn build_dyn_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    dyn_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    if let ty::Dynamic(..) = dyn_type.kind() {
        let type_name = compute_debuginfo_type_name(cx.tcx, dyn_type, true);
        type_map::build_type_with_children(
            cx,
            type_map::stub(
                cx,
                Stub::Struct,
                unique_type_id,
                &type_name,
                cx.size_and_align_of(dyn_type),
                NO_SCOPE_METADATA,
                DIFlags::FlagZero,
            ),
            |_, _| smallvec![],
            NO_GENERICS,
        )
    } else {
        bug!(
            "Only ty::Dynamic is valid for build_dyn_type_di_node(). Found {:?} instead.",
            dyn_type
        )
    }
}

/// Create debuginfo for `[T]` and `str`. These are unsized.
///
/// NOTE: We currently emit just emit the debuginfo for the element type here
/// (i.e. `T` for slices and `u8` for `str`), so that we end up with
/// `*const T` for the `data_ptr` field of the corresponding fat-pointer
/// debuginfo of `&[T]`.
///
/// It would be preferable and more accurate if we emitted a DIArray of T
/// without an upper bound instead. That is, LLVM already supports emitting
/// debuginfo of arrays of unknown size. But GDB currently seems to end up
/// in an infinite loop when confronted with such a type.
///
/// As a side effect of the current encoding every instance of a type like
/// `struct Foo { unsized_field: [u8] }` will look like
/// `struct Foo { unsized_field: u8 }` in debuginfo. If the length of the
/// slice is zero, then accessing `unsized_field` in the debugger would
/// result in an out-of-bounds access.
fn build_slice_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    slice_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let element_type = match slice_type.kind() {
        ty::Slice(element_type) => *element_type,
        ty::Str => cx.tcx.types.u8,
        _ => {
            bug!(
                "Only ty::Slice is valid for build_slice_type_di_node(). Found {:?} instead.",
                slice_type
            )
        }
    };

    let element_type_di_node = type_di_node(cx, element_type);
    return_if_di_node_created_in_meantime!(cx, unique_type_id);
    DINodeCreationResult { di_node: element_type_di_node, already_stored_in_typemap: false }
}

/// Get the debuginfo node for the given type.
///
/// This function will look up the debuginfo node in the TypeMap. If it can't find it, it
/// will create the node by dispatching to the corresponding `build_*_di_node()` function.
pub fn type_di_node<'ll, 'tcx>(cx: &CodegenCx<'ll, 'tcx>, t: Ty<'tcx>) -> &'ll DIType {
    let unique_type_id = UniqueTypeId::for_ty(cx.tcx, t);

    if let Some(existing_di_node) = debug_context(cx).type_map.di_node_for_unique_id(unique_type_id)
    {
        return existing_di_node;
    }

    debug!("type_di_node: {:?} kind: {:?}", t, t.kind());

    let DINodeCreationResult { di_node, already_stored_in_typemap } = match *t.kind() {
        ty::Never | ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) => {
            build_basic_type_di_node(cx, t)
        }
        ty::Tuple(elements) if elements.is_empty() => build_basic_type_di_node(cx, t),
        ty::Array(..) => build_fixed_size_array_di_node(cx, unique_type_id, t),
        ty::Slice(_) | ty::Str => build_slice_type_di_node(cx, t, unique_type_id),
        ty::Dynamic(..) => build_dyn_type_di_node(cx, t, unique_type_id),
        ty::Foreign(..) => build_foreign_type_di_node(cx, t, unique_type_id),
        ty::RawPtr(pointee_type, _) | ty::Ref(_, pointee_type, _) => {
            build_pointer_or_reference_di_node(cx, t, pointee_type, unique_type_id)
        }
        // Some `Box` are newtyped pointers, make debuginfo aware of that.
        // Only works if the allocator argument is a 1-ZST and hence irrelevant for layout
        // (or if there is no allocator argument).
        ty::Adt(def, args)
            if def.is_box()
                && args.get(1).map_or(true, |arg| cx.layout_of(arg.expect_ty()).is_1zst()) =>
        {
            build_pointer_or_reference_di_node(cx, t, t.boxed_ty(), unique_type_id)
        }
        ty::FnDef(..) | ty::FnPtr(_) => build_subroutine_type_di_node(cx, unique_type_id),
        ty::Closure(..) => build_closure_env_di_node(cx, unique_type_id),
        ty::CoroutineClosure(..) => build_closure_env_di_node(cx, unique_type_id),
        ty::Coroutine(..) => enums::build_coroutine_di_node(cx, unique_type_id),
        ty::Adt(def, ..) => match def.adt_kind() {
            AdtKind::Struct => build_struct_type_di_node(cx, unique_type_id),
            AdtKind::Union => build_union_type_di_node(cx, unique_type_id),
            AdtKind::Enum => enums::build_enum_type_di_node(cx, unique_type_id),
        },
        ty::Tuple(_) => build_tuple_type_di_node(cx, unique_type_id),
        // Type parameters from polymorphized functions.
        ty::Param(_) => build_param_type_di_node(cx, t),
        _ => bug!("debuginfo: unexpected type in type_di_node(): {:?}", t),
    };

    {
        if already_stored_in_typemap {
            // Make sure that we really do have a `TypeMap` entry for the unique type ID.
            let di_node_for_uid =
                match debug_context(cx).type_map.di_node_for_unique_id(unique_type_id) {
                    Some(di_node) => di_node,
                    None => {
                        bug!(
                            "expected type debuginfo node for unique \
                               type ID '{:?}' to already be in \
                               the `debuginfo::TypeMap` but it \
                               was not.",
                            unique_type_id,
                        );
                    }
                };

            assert_eq!(di_node_for_uid as *const _, di_node as *const _);
        } else {
            debug_context(cx).type_map.insert(unique_type_id, di_node);
        }
    }

    di_node
}

// FIXME(mw): Cache this via a regular UniqueTypeId instead of an extra field in the debug context.
fn recursion_marker_type_di_node<'ll, 'tcx>(cx: &CodegenCx<'ll, 'tcx>) -> &'ll DIType {
    *debug_context(cx).recursion_marker_type.get_or_init(move || {
        unsafe {
            // The choice of type here is pretty arbitrary -
            // anything reading the debuginfo for a recursive
            // type is going to see *something* weird - the only
            // question is what exactly it will see.
            //
            // FIXME: the name `<recur_type>` does not fit the naming scheme
            //        of other types.
            //
            // FIXME: it might make sense to use an actual pointer type here
            //        so that debuggers can show the address.
            let name = "<recur_type>";
            llvm::LLVMRustDIBuilderCreateBasicType(
                DIB(cx),
                name.as_ptr().cast(),
                name.len(),
                cx.tcx.data_layout.pointer_size.bits(),
                DW_ATE_unsigned,
            )
        }
    })
}

fn hex_encode(data: &[u8]) -> String {
    let mut hex_string = String::with_capacity(data.len() * 2);
    for byte in data.iter() {
        write!(&mut hex_string, "{byte:02x}").unwrap();
    }
    hex_string
}

pub fn file_metadata<'ll>(cx: &CodegenCx<'ll, '_>, source_file: &SourceFile) -> &'ll DIFile {
    let cache_key = Some((source_file.stable_id, source_file.src_hash));
    return debug_context(cx)
        .created_files
        .borrow_mut()
        .entry(cache_key)
        .or_insert_with(|| alloc_new_file_metadata(cx, source_file));

    #[instrument(skip(cx, source_file), level = "debug")]
    fn alloc_new_file_metadata<'ll>(
        cx: &CodegenCx<'ll, '_>,
        source_file: &SourceFile,
    ) -> &'ll DIFile {
        debug!(?source_file.name);

        let filename_display_preference =
            cx.sess().filename_display_preference(RemapPathScopeComponents::DEBUGINFO);

        use rustc_session::config::RemapPathScopeComponents;
        let (directory, file_name) = match &source_file.name {
            FileName::Real(filename) => {
                let working_directory = &cx.sess().opts.working_dir;
                debug!(?working_directory);

                if filename_display_preference == FileNameDisplayPreference::Remapped {
                    let filename = cx
                        .sess()
                        .source_map()
                        .path_mapping()
                        .to_embeddable_absolute_path(filename.clone(), working_directory);

                    // Construct the absolute path of the file
                    let abs_path = filename.remapped_path_if_available();
                    debug!(?abs_path);

                    if let Ok(rel_path) =
                        abs_path.strip_prefix(working_directory.remapped_path_if_available())
                    {
                        // If the compiler's working directory (which also is the DW_AT_comp_dir of
                        // the compilation unit) is a prefix of the path we are about to emit, then
                        // only emit the part relative to the working directory.
                        // Because of path remapping we sometimes see strange things here: `abs_path`
                        // might actually look like a relative path
                        // (e.g. `<crate-name-and-version>/src/lib.rs`), so if we emit it without
                        // taking the working directory into account, downstream tooling will
                        // interpret it as `<working-directory>/<crate-name-and-version>/src/lib.rs`,
                        // which makes no sense. Usually in such cases the working directory will also
                        // be remapped to `<crate-name-and-version>` or some other prefix of the path
                        // we are remapping, so we end up with
                        // `<crate-name-and-version>/<crate-name-and-version>/src/lib.rs`.
                        // By moving the working directory portion into the `directory` part of the
                        // DIFile, we allow LLVM to emit just the relative path for DWARF, while
                        // still emitting the correct absolute path for CodeView.
                        (
                            working_directory.to_string_lossy(FileNameDisplayPreference::Remapped),
                            rel_path.to_string_lossy().into_owned(),
                        )
                    } else {
                        ("".into(), abs_path.to_string_lossy().into_owned())
                    }
                } else {
                    let working_directory = working_directory.local_path_if_available();
                    let filename = filename.local_path_if_available();

                    debug!(?working_directory, ?filename);

                    let abs_path: Cow<'_, Path> = if filename.is_absolute() {
                        filename.into()
                    } else {
                        let mut p = PathBuf::new();
                        p.push(working_directory);
                        p.push(filename);
                        p.into()
                    };

                    if let Ok(rel_path) = abs_path.strip_prefix(working_directory) {
                        (
                            working_directory.to_string_lossy(),
                            rel_path.to_string_lossy().into_owned(),
                        )
                    } else {
                        ("".into(), abs_path.to_string_lossy().into_owned())
                    }
                }
            }
            other => {
                debug!(?other);
                ("".into(), other.display(filename_display_preference).to_string())
            }
        };

        let hash_kind = match source_file.src_hash.kind {
            rustc_span::SourceFileHashAlgorithm::Md5 => llvm::ChecksumKind::MD5,
            rustc_span::SourceFileHashAlgorithm::Sha1 => llvm::ChecksumKind::SHA1,
            rustc_span::SourceFileHashAlgorithm::Sha256 => llvm::ChecksumKind::SHA256,
        };
        let hash_value = hex_encode(source_file.src_hash.hash_bytes());

        unsafe {
            llvm::LLVMRustDIBuilderCreateFile(
                DIB(cx),
                file_name.as_ptr().cast(),
                file_name.len(),
                directory.as_ptr().cast(),
                directory.len(),
                hash_kind,
                hash_value.as_ptr().cast(),
                hash_value.len(),
            )
        }
    }
}

pub fn unknown_file_metadata<'ll>(cx: &CodegenCx<'ll, '_>) -> &'ll DIFile {
    debug_context(cx).created_files.borrow_mut().entry(None).or_insert_with(|| unsafe {
        let file_name = "<unknown>";
        let directory = "";
        let hash_value = "";

        llvm::LLVMRustDIBuilderCreateFile(
            DIB(cx),
            file_name.as_ptr().cast(),
            file_name.len(),
            directory.as_ptr().cast(),
            directory.len(),
            llvm::ChecksumKind::None,
            hash_value.as_ptr().cast(),
            hash_value.len(),
        )
    })
}

trait MsvcBasicName {
    fn msvc_basic_name(self) -> &'static str;
}

impl MsvcBasicName for ty::IntTy {
    fn msvc_basic_name(self) -> &'static str {
        match self {
            ty::IntTy::Isize => "ptrdiff_t",
            ty::IntTy::I8 => "__int8",
            ty::IntTy::I16 => "__int16",
            ty::IntTy::I32 => "__int32",
            ty::IntTy::I64 => "__int64",
            ty::IntTy::I128 => "__int128",
        }
    }
}

impl MsvcBasicName for ty::UintTy {
    fn msvc_basic_name(self) -> &'static str {
        match self {
            ty::UintTy::Usize => "size_t",
            ty::UintTy::U8 => "unsigned __int8",
            ty::UintTy::U16 => "unsigned __int16",
            ty::UintTy::U32 => "unsigned __int32",
            ty::UintTy::U64 => "unsigned __int64",
            ty::UintTy::U128 => "unsigned __int128",
        }
    }
}

impl MsvcBasicName for ty::FloatTy {
    fn msvc_basic_name(self) -> &'static str {
        // FIXME(f16_f128): `f16` and `f128` have no MSVC representation. We could improve the
        // debuginfo. See: <https://github.com/rust-lang/rust/issues/121837>
        match self {
            ty::FloatTy::F16 => {
                bug!("`f16` should have been handled in `build_basic_type_di_node`")
            }
            ty::FloatTy::F32 => "float",
            ty::FloatTy::F64 => "double",
            ty::FloatTy::F128 => "fp128",
        }
    }
}

fn build_cpp_f16_di_node<'ll, 'tcx>(cx: &CodegenCx<'ll, 'tcx>) -> DINodeCreationResult<'ll> {
    // MSVC has no native support for `f16`. Instead, emit `struct f16 { bits: u16 }` to allow the
    // `f16`'s value to be displayed using a Natvis visualiser in `intrinsic.natvis`.
    let float_ty = cx.tcx.types.f16;
    let bits_ty = cx.tcx.types.u16;
    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            UniqueTypeId::for_ty(cx.tcx, float_ty),
            "f16",
            cx.size_and_align_of(float_ty),
            NO_SCOPE_METADATA,
            DIFlags::FlagZero,
        ),
        // Fields:
        |cx, float_di_node| {
            smallvec![build_field_di_node(
                cx,
                float_di_node,
                "bits",
                cx.size_and_align_of(bits_ty),
                Size::ZERO,
                DIFlags::FlagZero,
                type_di_node(cx, bits_ty),
            )]
        },
        NO_GENERICS,
    )
}

fn build_basic_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    t: Ty<'tcx>,
) -> DINodeCreationResult<'ll> {
    debug!("build_basic_type_di_node: {:?}", t);

    // When targeting MSVC, emit MSVC style type names for compatibility with
    // .natvis visualizers (and perhaps other existing native debuggers?)
    let cpp_like_debuginfo = cpp_like_debuginfo(cx.tcx);

    let (name, encoding) = match t.kind() {
        ty::Never => ("!", DW_ATE_unsigned),
        ty::Tuple(elements) if elements.is_empty() => {
            if cpp_like_debuginfo {
                return build_tuple_type_di_node(cx, UniqueTypeId::for_ty(cx.tcx, t));
            } else {
                ("()", DW_ATE_unsigned)
            }
        }
        ty::Bool => ("bool", DW_ATE_boolean),
        ty::Char => ("char", DW_ATE_UTF),
        ty::Int(int_ty) if cpp_like_debuginfo => (int_ty.msvc_basic_name(), DW_ATE_signed),
        ty::Uint(uint_ty) if cpp_like_debuginfo => (uint_ty.msvc_basic_name(), DW_ATE_unsigned),
        ty::Float(ty::FloatTy::F16) if cpp_like_debuginfo => {
            return build_cpp_f16_di_node(cx);
        }
        ty::Float(float_ty) if cpp_like_debuginfo => (float_ty.msvc_basic_name(), DW_ATE_float),
        ty::Int(int_ty) => (int_ty.name_str(), DW_ATE_signed),
        ty::Uint(uint_ty) => (uint_ty.name_str(), DW_ATE_unsigned),
        ty::Float(float_ty) => (float_ty.name_str(), DW_ATE_float),
        _ => bug!("debuginfo::build_basic_type_di_node - `t` is invalid type"),
    };

    let ty_di_node = unsafe {
        llvm::LLVMRustDIBuilderCreateBasicType(
            DIB(cx),
            name.as_ptr().cast(),
            name.len(),
            cx.size_of(t).bits(),
            encoding,
        )
    };

    if !cpp_like_debuginfo {
        return DINodeCreationResult::new(ty_di_node, false);
    }

    let typedef_name = match t.kind() {
        ty::Int(int_ty) => int_ty.name_str(),
        ty::Uint(uint_ty) => uint_ty.name_str(),
        ty::Float(float_ty) => float_ty.name_str(),
        _ => return DINodeCreationResult::new(ty_di_node, false),
    };

    let typedef_di_node = unsafe {
        llvm::LLVMRustDIBuilderCreateTypedef(
            DIB(cx),
            ty_di_node,
            typedef_name.as_ptr().cast(),
            typedef_name.len(),
            unknown_file_metadata(cx),
            0,
            None,
        )
    };

    DINodeCreationResult::new(typedef_di_node, false)
}

fn build_foreign_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    t: Ty<'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    debug!("build_foreign_type_di_node: {:?}", t);

    let &ty::Foreign(def_id) = unique_type_id.expect_ty().kind() else {
        bug!(
            "build_foreign_type_di_node() called with unexpected type: {:?}",
            unique_type_id.expect_ty()
        );
    };

    build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            unique_type_id,
            &compute_debuginfo_type_name(cx.tcx, t, false),
            cx.size_and_align_of(t),
            Some(get_namespace_for_item(cx, def_id)),
            DIFlags::FlagZero,
        ),
        |_, _| smallvec![],
        NO_GENERICS,
    )
}

fn build_param_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    t: Ty<'tcx>,
) -> DINodeCreationResult<'ll> {
    debug!("build_param_type_di_node: {:?}", t);
    let name = format!("{t:?}");
    DINodeCreationResult {
        di_node: unsafe {
            llvm::LLVMRustDIBuilderCreateBasicType(
                DIB(cx),
                name.as_ptr().cast(),
                name.len(),
                Size::ZERO.bits(),
                DW_ATE_unsigned,
            )
        },
        already_stored_in_typemap: false,
    }
}

pub fn build_compile_unit_di_node<'ll, 'tcx>(
    tcx: TyCtxt<'tcx>,
    codegen_unit_name: &str,
    debug_context: &CodegenUnitDebugContext<'ll, 'tcx>,
) -> &'ll DIDescriptor {
    use rustc_session::{config::RemapPathScopeComponents, RemapFileNameExt};
    let mut name_in_debuginfo = tcx
        .sess
        .local_crate_source_file()
        .map(|src| src.for_scope(&tcx.sess, RemapPathScopeComponents::DEBUGINFO).to_path_buf())
        .unwrap_or_else(|| PathBuf::from(tcx.crate_name(LOCAL_CRATE).as_str()));

    // To avoid breaking split DWARF, we need to ensure that each codegen unit
    // has a unique `DW_AT_name`. This is because there's a remote chance that
    // different codegen units for the same module will have entirely
    // identical DWARF entries for the purpose of the DWO ID, which would
    // violate Appendix F ("Split Dwarf Object Files") of the DWARF 5
    // specification. LLVM uses the algorithm specified in section 7.32 "Type
    // Signature Computation" to compute the DWO ID, which does not include
    // any fields that would distinguish compilation units. So we must embed
    // the codegen unit name into the `DW_AT_name`. (Issue #88521.)
    //
    // Additionally, the OSX linker has an idiosyncrasy where it will ignore
    // some debuginfo if multiple object files with the same `DW_AT_name` are
    // linked together.
    //
    // As a workaround for these two issues, we generate unique names for each
    // object file. Those do not correspond to an actual source file but that
    // is harmless.
    name_in_debuginfo.push("@");
    name_in_debuginfo.push(codegen_unit_name);

    debug!("build_compile_unit_di_node: {:?}", name_in_debuginfo);
    let rustc_producer = format!("rustc version {}", tcx.sess.cfg_version);
    // FIXME(#41252) Remove "clang LLVM" if we can get GDB and LLVM to play nice.
    let producer = format!("clang LLVM ({rustc_producer})");

    let name_in_debuginfo = name_in_debuginfo.to_string_lossy();
    let work_dir = tcx
        .sess
        .opts
        .working_dir
        .for_scope(tcx.sess, RemapPathScopeComponents::DEBUGINFO)
        .to_string_lossy();
    let output_filenames = tcx.output_filenames(());
    let split_name = if tcx.sess.target_can_use_split_dwarf()
        && let Some(f) = output_filenames.split_dwarf_path(
            tcx.sess.split_debuginfo(),
            tcx.sess.opts.unstable_opts.split_dwarf_kind,
            Some(codegen_unit_name),
        ) {
        // We get a path relative to the working directory from split_dwarf_path
        Some(tcx.sess.source_map().path_mapping().to_real_filename(f))
    } else {
        None
    };
    let split_name = split_name
        .as_ref()
        .map(|f| f.for_scope(tcx.sess, RemapPathScopeComponents::DEBUGINFO).to_string_lossy())
        .unwrap_or_default();
    let kind = DebugEmissionKind::from_generic(tcx.sess.opts.debuginfo);

    let dwarf_version =
        tcx.sess.opts.unstable_opts.dwarf_version.unwrap_or(tcx.sess.target.default_dwarf_version);
    let is_dwarf_kind =
        matches!(tcx.sess.target.debuginfo_kind, DebuginfoKind::Dwarf | DebuginfoKind::DwarfDsym);
    // Don't emit `.debug_pubnames` and `.debug_pubtypes` on DWARFv4 or lower.
    let debug_name_table_kind = if is_dwarf_kind && dwarf_version <= 4 {
        DebugNameTableKind::None
    } else {
        DebugNameTableKind::Default
    };

    unsafe {
        let compile_unit_file = llvm::LLVMRustDIBuilderCreateFile(
            debug_context.builder,
            name_in_debuginfo.as_ptr().cast(),
            name_in_debuginfo.len(),
            work_dir.as_ptr().cast(),
            work_dir.len(),
            llvm::ChecksumKind::None,
            ptr::null(),
            0,
        );

        let unit_metadata = llvm::LLVMRustDIBuilderCreateCompileUnit(
            debug_context.builder,
            DW_LANG_RUST,
            compile_unit_file,
            producer.as_ptr().cast(),
            producer.len(),
            tcx.sess.opts.optimize != config::OptLevel::No,
            c"".as_ptr().cast(),
            0,
            // NB: this doesn't actually have any perceptible effect, it seems. LLVM will instead
            // put the path supplied to `MCSplitDwarfFile` into the debug info of the final
            // output(s).
            split_name.as_ptr().cast(),
            split_name.len(),
            kind,
            0,
            tcx.sess.opts.unstable_opts.split_dwarf_inlining,
            debug_name_table_kind,
        );

        if tcx.sess.opts.unstable_opts.profile {
            let default_gcda_path = &output_filenames.with_extension("gcda");
            let gcda_path =
                tcx.sess.opts.unstable_opts.profile_emit.as_ref().unwrap_or(default_gcda_path);

            let gcov_cu_info = [
                path_to_mdstring(debug_context.llcontext, &output_filenames.with_extension("gcno")),
                path_to_mdstring(debug_context.llcontext, gcda_path),
                unit_metadata,
            ];
            let gcov_metadata = llvm::LLVMMDNodeInContext2(
                debug_context.llcontext,
                gcov_cu_info.as_ptr(),
                gcov_cu_info.len(),
            );
            let val = llvm::LLVMMetadataAsValue(debug_context.llcontext, gcov_metadata);

            llvm::LLVMAddNamedMetadataOperand(debug_context.llmod, c"llvm.gcov".as_ptr(), val);
        }

        return unit_metadata;
    };

    fn path_to_mdstring<'ll>(llcx: &'ll llvm::Context, path: &Path) -> &'ll llvm::Metadata {
        let path_str = path_to_c_string(path);
        unsafe { llvm::LLVMMDStringInContext2(llcx, path_str.as_ptr(), path_str.as_bytes().len()) }
    }
}

/// Creates a `DW_TAG_member` entry inside the DIE represented by the given `type_di_node`.
fn build_field_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    owner: &'ll DIScope,
    name: &str,
    size_and_align: (Size, Align),
    offset: Size,
    flags: DIFlags,
    type_di_node: &'ll DIType,
) -> &'ll DIType {
    unsafe {
        llvm::LLVMRustDIBuilderCreateMemberType(
            DIB(cx),
            owner,
            name.as_ptr().cast(),
            name.len(),
            unknown_file_metadata(cx),
            UNKNOWN_LINE_NUMBER,
            size_and_align.0.bits(),
            size_and_align.1.bits() as u32,
            offset.bits(),
            flags,
            type_di_node,
        )
    }
}

/// Returns the `DIFlags` corresponding to the visibility of the item identified by `did`.
///
/// `DIFlags::Flag{Public,Protected,Private}` correspond to `DW_AT_accessibility`
/// (public/protected/private) aren't exactly right for Rust, but neither is `DW_AT_visibility`
/// (local/exported/qualified), and there's no way to set `DW_AT_visibility` in LLVM's API.
fn visibility_di_flags<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    did: DefId,
    type_did: DefId,
) -> DIFlags {
    let parent_did = cx.tcx.parent(type_did);
    let visibility = cx.tcx.visibility(did);
    match visibility {
        Visibility::Public => DIFlags::FlagPublic,
        // Private fields have a restricted visibility of the module containing the type.
        Visibility::Restricted(did) if did == parent_did => DIFlags::FlagPrivate,
        // `pub(crate)`/`pub(super)` visibilities are any other restricted visibility.
        Visibility::Restricted(..) => DIFlags::FlagProtected,
    }
}

/// Creates the debuginfo node for a Rust struct type. Maybe be a regular struct or a tuple-struct.
fn build_struct_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let struct_type = unique_type_id.expect_ty();
    let ty::Adt(adt_def, _) = struct_type.kind() else {
        bug!("build_struct_type_di_node() called with non-struct-type: {:?}", struct_type);
    };
    assert!(adt_def.is_struct());
    let containing_scope = get_namespace_for_item(cx, adt_def.did());
    let struct_type_and_layout = cx.layout_of(struct_type);
    let variant_def = adt_def.non_enum_variant();

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            unique_type_id,
            &compute_debuginfo_type_name(cx.tcx, struct_type, false),
            size_and_align_of(struct_type_and_layout),
            Some(containing_scope),
            visibility_di_flags(cx, adt_def.did(), adt_def.did()),
        ),
        // Fields:
        |cx, owner| {
            variant_def
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let field_name = if variant_def.ctor_kind() == Some(CtorKind::Fn) {
                        // This is a tuple struct
                        tuple_field_name(i)
                    } else {
                        // This is struct with named fields
                        Cow::Borrowed(f.name.as_str())
                    };
                    let field_layout = struct_type_and_layout.field(cx, i);
                    build_field_di_node(
                        cx,
                        owner,
                        &field_name[..],
                        (field_layout.size, field_layout.align.abi),
                        struct_type_and_layout.fields.offset(i),
                        visibility_di_flags(cx, f.did, adt_def.did()),
                        type_di_node(cx, field_layout.ty),
                    )
                })
                .collect()
        },
        |cx| build_generic_type_param_di_nodes(cx, struct_type),
    )
}

//=-----------------------------------------------------------------------------
// Tuples
//=-----------------------------------------------------------------------------

/// Builds the DW_TAG_member debuginfo nodes for the upvars of a closure or coroutine.
/// For a coroutine, this will handle upvars shared by all states.
fn build_upvar_field_di_nodes<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    closure_or_coroutine_ty: Ty<'tcx>,
    closure_or_coroutine_di_node: &'ll DIType,
) -> SmallVec<&'ll DIType> {
    let (&def_id, up_var_tys) = match closure_or_coroutine_ty.kind() {
        ty::Coroutine(def_id, args) => (def_id, args.as_coroutine().prefix_tys()),
        ty::Closure(def_id, args) => (def_id, args.as_closure().upvar_tys()),
        ty::CoroutineClosure(def_id, args) => (def_id, args.as_coroutine_closure().upvar_tys()),
        _ => {
            bug!(
                "build_upvar_field_di_nodes() called with non-closure-or-coroutine-type: {:?}",
                closure_or_coroutine_ty
            )
        }
    };

    assert!(
        up_var_tys.iter().all(|t| t == cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), t))
    );

    let capture_names = cx.tcx.closure_saved_names_of_captured_variables(def_id);
    let layout = cx.layout_of(closure_or_coroutine_ty);

    up_var_tys
        .into_iter()
        .zip(capture_names.iter())
        .enumerate()
        .map(|(index, (up_var_ty, capture_name))| {
            build_field_di_node(
                cx,
                closure_or_coroutine_di_node,
                capture_name.as_str(),
                cx.size_and_align_of(up_var_ty),
                layout.fields.offset(index),
                DIFlags::FlagZero,
                type_di_node(cx, up_var_ty),
            )
        })
        .collect()
}

/// Builds the DW_TAG_structure_type debuginfo node for a Rust tuple type.
fn build_tuple_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let tuple_type = unique_type_id.expect_ty();
    let &ty::Tuple(component_types) = tuple_type.kind() else {
        bug!("build_tuple_type_di_node() called with non-tuple-type: {:?}", tuple_type)
    };

    let tuple_type_and_layout = cx.layout_of(tuple_type);
    let type_name = compute_debuginfo_type_name(cx.tcx, tuple_type, false);

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            unique_type_id,
            &type_name,
            size_and_align_of(tuple_type_and_layout),
            NO_SCOPE_METADATA,
            DIFlags::FlagZero,
        ),
        // Fields:
        |cx, tuple_di_node| {
            component_types
                .into_iter()
                .enumerate()
                .map(|(index, component_type)| {
                    build_field_di_node(
                        cx,
                        tuple_di_node,
                        &tuple_field_name(index),
                        cx.size_and_align_of(component_type),
                        tuple_type_and_layout.fields.offset(index),
                        DIFlags::FlagZero,
                        type_di_node(cx, component_type),
                    )
                })
                .collect()
        },
        NO_GENERICS,
    )
}

/// Builds the debuginfo node for a closure environment.
fn build_closure_env_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let closure_env_type = unique_type_id.expect_ty();
    let &(ty::Closure(def_id, _) | ty::CoroutineClosure(def_id, _)) = closure_env_type.kind()
    else {
        bug!("build_closure_env_di_node() called with non-closure-type: {:?}", closure_env_type)
    };
    let containing_scope = get_namespace_for_item(cx, def_id);
    let type_name = compute_debuginfo_type_name(cx.tcx, closure_env_type, false);

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            unique_type_id,
            &type_name,
            cx.size_and_align_of(closure_env_type),
            Some(containing_scope),
            DIFlags::FlagZero,
        ),
        // Fields:
        |cx, owner| build_upvar_field_di_nodes(cx, closure_env_type, owner),
        NO_GENERICS,
    )
}

/// Build the debuginfo node for a Rust `union` type.
fn build_union_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let union_type = unique_type_id.expect_ty();
    let (union_def_id, variant_def) = match union_type.kind() {
        ty::Adt(def, _) => (def.did(), def.non_enum_variant()),
        _ => bug!("build_union_type_di_node on a non-ADT"),
    };
    let containing_scope = get_namespace_for_item(cx, union_def_id);
    let union_ty_and_layout = cx.layout_of(union_type);
    let type_name = compute_debuginfo_type_name(cx.tcx, union_type, false);

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Union,
            unique_type_id,
            &type_name,
            size_and_align_of(union_ty_and_layout),
            Some(containing_scope),
            DIFlags::FlagZero,
        ),
        // Fields:
        |cx, owner| {
            variant_def
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let field_layout = union_ty_and_layout.field(cx, i);
                    build_field_di_node(
                        cx,
                        owner,
                        f.name.as_str(),
                        size_and_align_of(field_layout),
                        Size::ZERO,
                        DIFlags::FlagZero,
                        type_di_node(cx, field_layout.ty),
                    )
                })
                .collect()
        },
        // Generics:
        |cx| build_generic_type_param_di_nodes(cx, union_type),
    )
}

/// Computes the type parameters for a type, if any, for the given metadata.
fn build_generic_type_param_di_nodes<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    ty: Ty<'tcx>,
) -> SmallVec<&'ll DIType> {
    if let ty::Adt(def, args) = *ty.kind() {
        if args.types().next().is_some() {
            let generics = cx.tcx.generics_of(def.did());
            let names = get_parameter_names(cx, generics);
            let template_params: SmallVec<_> = iter::zip(args, names)
                .filter_map(|(kind, name)| {
                    kind.as_type().map(|ty| {
                        let actual_type =
                            cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), ty);
                        let actual_type_di_node = type_di_node(cx, actual_type);
                        let name = name.as_str();
                        unsafe {
                            llvm::LLVMRustDIBuilderCreateTemplateTypeParameter(
                                DIB(cx),
                                None,
                                name.as_ptr().cast(),
                                name.len(),
                                actual_type_di_node,
                            )
                        }
                    })
                })
                .collect();

            return template_params;
        }
    }

    return smallvec![];

    fn get_parameter_names(cx: &CodegenCx<'_, '_>, generics: &ty::Generics) -> Vec<Symbol> {
        let mut names = generics
            .parent
            .map_or_else(Vec::new, |def_id| get_parameter_names(cx, cx.tcx.generics_of(def_id)));
        names.extend(generics.own_params.iter().map(|param| param.name));
        names
    }
}

/// Creates debug information for the given global variable.
///
/// Adds the created debuginfo nodes directly to the crate's IR.
pub fn build_global_var_di_node<'ll>(cx: &CodegenCx<'ll, '_>, def_id: DefId, global: &'ll Value) {
    if cx.dbg_cx.is_none() {
        return;
    }

    // Only create type information if full debuginfo is enabled
    if cx.sess().opts.debuginfo != DebugInfo::Full {
        return;
    }

    let tcx = cx.tcx;

    // We may want to remove the namespace scope if we're in an extern block (see
    // https://github.com/rust-lang/rust/pull/46457#issuecomment-351750952).
    let var_scope = get_namespace_for_item(cx, def_id);
    let span = hygiene::walk_chain_collapsed(tcx.def_span(def_id), DUMMY_SP);

    let (file_metadata, line_number) = if !span.is_dummy() {
        let loc = cx.lookup_debug_loc(span.lo());
        (file_metadata(cx, &loc.file), loc.line)
    } else {
        (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER)
    };

    let is_local_to_unit = is_node_local_to_unit(cx, def_id);

    let DefKind::Static { nested, .. } = cx.tcx.def_kind(def_id) else { bug!() };
    if nested {
        return;
    }
    let variable_type = Instance::mono(cx.tcx, def_id).ty(cx.tcx, ty::ParamEnv::reveal_all());
    let type_di_node = type_di_node(cx, variable_type);
    let var_name = tcx.item_name(def_id);
    let var_name = var_name.as_str();
    let linkage_name = mangled_name_of_instance(cx, Instance::mono(tcx, def_id)).name;
    // When empty, linkage_name field is omitted,
    // which is what we want for no_mangle statics
    let linkage_name = if var_name == linkage_name { "" } else { linkage_name };

    let global_align = cx.align_of(variable_type);

    unsafe {
        llvm::LLVMRustDIBuilderCreateStaticVariable(
            DIB(cx),
            Some(var_scope),
            var_name.as_ptr().cast(),
            var_name.len(),
            linkage_name.as_ptr().cast(),
            linkage_name.len(),
            file_metadata,
            line_number,
            type_di_node,
            is_local_to_unit,
            global,
            None,
            global_align.bits() as u32,
        );
    }
}

/// Generates LLVM debuginfo for a vtable.
///
/// The vtable type looks like a struct with a field for each function pointer and super-trait
/// pointer it contains (plus the `size` and `align` fields).
///
/// Except for `size`, `align`, and `drop_in_place`, the field names don't try to mirror
/// the name of the method they implement. This can be implemented in the future once there
/// is a proper disambiguation scheme for dealing with methods from different traits that have
/// the same name.
fn build_vtable_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    ty: Ty<'tcx>,
    poly_trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
) -> &'ll DIType {
    let tcx = cx.tcx;

    let vtable_entries = if let Some(poly_trait_ref) = poly_trait_ref {
        let trait_ref = poly_trait_ref.with_self_ty(tcx, ty);
        let trait_ref = tcx.erase_regions(trait_ref);

        tcx.vtable_entries(trait_ref)
    } else {
        TyCtxt::COMMON_VTABLE_ENTRIES
    };

    // All function pointers are described as opaque pointers. This could be improved in the future
    // by describing them as actual function pointers.
    let void_pointer_ty = Ty::new_imm_ptr(tcx, tcx.types.unit);
    let void_pointer_type_di_node = type_di_node(cx, void_pointer_ty);
    let usize_di_node = type_di_node(cx, tcx.types.usize);
    let (pointer_size, pointer_align) = cx.size_and_align_of(void_pointer_ty);
    // If `usize` is not pointer-sized and -aligned then the size and alignment computations
    // for the vtable as a whole would be wrong. Let's make sure this holds even on weird
    // platforms.
    assert_eq!(cx.size_and_align_of(tcx.types.usize), (pointer_size, pointer_align));

    let vtable_type_name =
        compute_debuginfo_vtable_name(cx.tcx, ty, poly_trait_ref, VTableNameKind::Type);
    let unique_type_id = UniqueTypeId::for_vtable_ty(tcx, ty, poly_trait_ref);
    let size = pointer_size * vtable_entries.len() as u64;

    // This gets mapped to a DW_AT_containing_type attribute which allows GDB to correlate
    // the vtable to the type it is for.
    let vtable_holder = type_di_node(cx, ty);

    build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::VTableTy { vtable_holder },
            unique_type_id,
            &vtable_type_name,
            (size, pointer_align),
            NO_SCOPE_METADATA,
            DIFlags::FlagArtificial,
        ),
        |cx, vtable_type_di_node| {
            vtable_entries
                .iter()
                .enumerate()
                .filter_map(|(index, vtable_entry)| {
                    let (field_name, field_type_di_node) = match vtable_entry {
                        ty::VtblEntry::MetadataDropInPlace => {
                            ("drop_in_place".to_string(), void_pointer_type_di_node)
                        }
                        ty::VtblEntry::Method(_) => {
                            // Note: This code does not try to give a proper name to each method
                            //       because their might be multiple methods with the same name
                            //       (coming from different traits).
                            (format!("__method{index}"), void_pointer_type_di_node)
                        }
                        ty::VtblEntry::TraitVPtr(_) => {
                            (format!("__super_trait_ptr{index}"), void_pointer_type_di_node)
                        }
                        ty::VtblEntry::MetadataAlign => ("align".to_string(), usize_di_node),
                        ty::VtblEntry::MetadataSize => ("size".to_string(), usize_di_node),
                        ty::VtblEntry::Vacant => return None,
                    };

                    let field_offset = pointer_size * index as u64;

                    Some(build_field_di_node(
                        cx,
                        vtable_type_di_node,
                        &field_name,
                        (pointer_size, pointer_align),
                        field_offset,
                        DIFlags::FlagZero,
                        field_type_di_node,
                    ))
                })
                .collect()
        },
        NO_GENERICS,
    )
    .di_node
}

pub(crate) fn apply_vcall_visibility_metadata<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    ty: Ty<'tcx>,
    trait_ref: Option<PolyExistentialTraitRef<'tcx>>,
    vtable: &'ll Value,
) {
    // FIXME(flip1995): The virtual function elimination optimization only works with full LTO in
    // LLVM at the moment.
    if !cx.sess().opts.unstable_opts.virtual_function_elimination || cx.sess().lto() != Lto::Fat {
        return;
    }

    enum VCallVisibility {
        Public = 0,
        LinkageUnit = 1,
        TranslationUnit = 2,
    }

    let Some(trait_ref) = trait_ref else { return };

    let trait_ref_self = trait_ref.with_self_ty(cx.tcx, ty);
    let trait_ref_self = cx.tcx.erase_regions(trait_ref_self);
    let trait_def_id = trait_ref_self.def_id();
    let trait_vis = cx.tcx.visibility(trait_def_id);

    let cgus = cx.sess().codegen_units().as_usize();
    let single_cgu = cgus == 1;

    let lto = cx.sess().lto();

    // Since LLVM requires full LTO for the virtual function elimination optimization to apply,
    // only the `Lto::Fat` cases are relevant currently.
    let vcall_visibility = match (lto, trait_vis, single_cgu) {
        // If there is not LTO and the visibility in public, we have to assume that the vtable can
        // be seen from anywhere. With multiple CGUs, the vtable is quasi-public.
        (Lto::No | Lto::ThinLocal, Visibility::Public, _)
        | (Lto::No, Visibility::Restricted(_), false) => VCallVisibility::Public,
        // With LTO and a quasi-public visibility, the usages of the functions of the vtable are
        // all known by the `LinkageUnit`.
        // FIXME: LLVM only supports this optimization for `Lto::Fat` currently. Once it also
        // supports `Lto::Thin` the `VCallVisibility` may have to be adjusted for those.
        (Lto::Fat | Lto::Thin, Visibility::Public, _)
        | (Lto::ThinLocal | Lto::Thin | Lto::Fat, Visibility::Restricted(_), false) => {
            VCallVisibility::LinkageUnit
        }
        // If there is only one CGU, private vtables can only be seen by that CGU/translation unit
        // and therefore we know of all usages of functions in the vtable.
        (_, Visibility::Restricted(_), true) => VCallVisibility::TranslationUnit,
    };

    let trait_ref_typeid = typeid_for_trait_ref(cx.tcx, trait_ref);

    unsafe {
        let typeid = llvm::LLVMMDStringInContext(
            cx.llcx,
            trait_ref_typeid.as_ptr() as *const c_char,
            trait_ref_typeid.as_bytes().len() as c_uint,
        );
        let v = [cx.const_usize(0), typeid];
        llvm::LLVMRustGlobalAddMetadata(
            vtable,
            llvm::MD_type as c_uint,
            llvm::LLVMValueAsMetadata(llvm::LLVMMDNodeInContext(
                cx.llcx,
                v.as_ptr(),
                v.len() as c_uint,
            )),
        );
        let vcall_visibility = llvm::LLVMValueAsMetadata(cx.const_u64(vcall_visibility as u64));
        let vcall_visibility_metadata = llvm::LLVMMDNodeInContext2(cx.llcx, &vcall_visibility, 1);
        llvm::LLVMGlobalSetMetadata(
            vtable,
            llvm::MetadataType::MD_vcall_visibility as c_uint,
            vcall_visibility_metadata,
        );
    }
}

/// Creates debug information for the given vtable, which is for the
/// given type.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_vtable_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    ty: Ty<'tcx>,
    poly_trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
    vtable: &'ll Value,
) {
    if cx.dbg_cx.is_none() {
        return;
    }

    // Only create type information if full debuginfo is enabled
    if cx.sess().opts.debuginfo != DebugInfo::Full {
        return;
    }

    // When full debuginfo is enabled, we want to try and prevent vtables from being
    // merged. Otherwise debuggers will have a hard time mapping from dyn pointer
    // to concrete type.
    llvm::SetUnnamedAddress(vtable, llvm::UnnamedAddr::No);

    let vtable_name =
        compute_debuginfo_vtable_name(cx.tcx, ty, poly_trait_ref, VTableNameKind::GlobalVariable);
    let vtable_type_di_node = build_vtable_type_di_node(cx, ty, poly_trait_ref);
    let linkage_name = "";

    unsafe {
        llvm::LLVMRustDIBuilderCreateStaticVariable(
            DIB(cx),
            NO_SCOPE_METADATA,
            vtable_name.as_ptr().cast(),
            vtable_name.len(),
            linkage_name.as_ptr().cast(),
            linkage_name.len(),
            unknown_file_metadata(cx),
            UNKNOWN_LINE_NUMBER,
            vtable_type_di_node,
            true,
            vtable,
            None,
            0,
        );
    }
}

/// Creates an "extension" of an existing `DIScope` into another file.
pub fn extend_scope_to_file<'ll>(
    cx: &CodegenCx<'ll, '_>,
    scope_metadata: &'ll DIScope,
    file: &SourceFile,
) -> &'ll DILexicalBlock {
    let file_metadata = file_metadata(cx, file);
    unsafe { llvm::LLVMRustDIBuilderCreateLexicalBlockFile(DIB(cx), scope_metadata, file_metadata) }
}

pub fn tuple_field_name(field_index: usize) -> Cow<'static, str> {
    const TUPLE_FIELD_NAMES: [&'static str; 16] = [
        "__0", "__1", "__2", "__3", "__4", "__5", "__6", "__7", "__8", "__9", "__10", "__11",
        "__12", "__13", "__14", "__15",
    ];
    TUPLE_FIELD_NAMES
        .get(field_index)
        .map(|s| Cow::from(*s))
        .unwrap_or_else(|| Cow::from(format!("__{field_index}")))
}
