use self::RecursiveTypeDescription::*;
use self::MemberDescriptionFactory::*;
use self::EnumDiscriminantInfo::*;

use super::utils::{debug_context, DIB, span_start,
                   get_namespace_for_item, create_DIArray, is_node_local_to_unit};
use super::namespace::mangled_name_of_instance;
use super::type_names::compute_debuginfo_type_name;
use super::{CrateDebugContext};
use crate::abi;
use crate::value::Value;
use rustc_codegen_ssa::traits::*;

use crate::llvm;
use crate::llvm::debuginfo::{DIArray, DIType, DIFile, DIScope, DIDescriptor,
                      DICompositeType, DILexicalBlock, DIFlags, DebugEmissionKind};
use crate::llvm_util;

use crate::common::CodegenCx;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc::hir::CodegenFnAttrFlags;
use rustc::hir::def::CtorKind;
use rustc::hir::def_id::{DefId, CrateNum, LOCAL_CRATE};
use rustc::ich::NodeIdHashingMode;
use rustc::mir::Field;
use rustc::mir::GeneratorLayout;
use rustc::mir::interpret::truncate;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc::ty::Instance;
use rustc::ty::{self, AdtKind, ParamEnv, Ty, TyCtxt};
use rustc::ty::layout::{self, Align, Integer, IntegerExt, LayoutOf,
                        PrimitiveExt, Size, TyLayout, VariantIdx};
use rustc::ty::subst::UnpackedKind;
use rustc::session::config::{self, DebugInfo};
use rustc::util::nodemap::FxHashMap;
use rustc_fs_util::path_to_c_string;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_target::abi::HasDataLayout;

use libc::{c_uint, c_longlong};
use std::collections::hash_map::Entry;
use std::ffi::CString;
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
use std::iter;
use std::ptr;
use std::path::{Path, PathBuf};
use syntax::ast;
use syntax::symbol::{Interner, InternedString};
use syntax_pos::{self, Span, FileName};

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
// See http://www.dwarfstd.org/ShowIssue.php?issue=140129.1
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
const DW_ATE_unsigned_char: c_uint = 0x08;

pub const UNKNOWN_LINE_NUMBER: c_uint = 0;
pub const UNKNOWN_COLUMN_NUMBER: c_uint = 0;

pub const NO_SCOPE_METADATA: Option<&DIScope> = None;

#[derive(Copy, Debug, Hash, Eq, PartialEq, Clone)]
pub struct UniqueTypeId(ast::Name);

// The TypeMap is where the CrateDebugContext holds the type metadata nodes
// created so far. The metadata nodes are indexed by UniqueTypeId, and, for
// faster lookup, also by Ty. The TypeMap is responsible for creating
// UniqueTypeIds.
#[derive(Default)]
pub struct TypeMap<'ll, 'tcx> {
    // The UniqueTypeIds created so far
    unique_id_interner: Interner,
    // A map from UniqueTypeId to debuginfo metadata for that type. This is a 1:1 mapping.
    unique_id_to_metadata: FxHashMap<UniqueTypeId, &'ll DIType>,
    // A map from types to debuginfo metadata. This is a N:1 mapping.
    type_to_metadata: FxHashMap<Ty<'tcx>, &'ll DIType>,
    // A map from types to UniqueTypeId. This is a N:1 mapping.
    type_to_unique_id: FxHashMap<Ty<'tcx>, UniqueTypeId>
}

impl TypeMap<'ll, 'tcx> {
    // Adds a Ty to metadata mapping to the TypeMap. The method will fail if
    // the mapping already exists.
    fn register_type_with_metadata(
        &mut self,
        type_: Ty<'tcx>,
        metadata: &'ll DIType,
    ) {
        if self.type_to_metadata.insert(type_, metadata).is_some() {
            bug!("Type metadata for Ty '{}' is already in the TypeMap!", type_);
        }
    }

    // Removes a Ty to metadata mapping
    // This is useful when computing the metadata for a potentially
    // recursive type (e.g. a function ptr of the form:
    //
    // fn foo() -> impl Copy { foo }
    //
    // This kind of type cannot be properly represented
    // via LLVM debuginfo. As a workaround,
    // we register a temporary Ty to metadata mapping
    // for the function before we compute its actual metadata.
    // If the metadata computation ends up recursing back to the
    // original function, it will use the temporary mapping
    // for the inner self-reference, preventing us from
    // recursing forever.
    //
    // This function is used to remove the temporary metadata
    // mapping after we've computed the actual metadata
    fn remove_type(
        &mut self,
        type_: Ty<'tcx>,
    ) {
        if self.type_to_metadata.remove(type_).is_none() {
            bug!("Type metadata Ty '{}' is not in the TypeMap!", type_);
        }
    }

    // Adds a UniqueTypeId to metadata mapping to the TypeMap. The method will
    // fail if the mapping already exists.
    fn register_unique_id_with_metadata(
        &mut self,
        unique_type_id: UniqueTypeId,
        metadata: &'ll DIType,
    ) {
        if self.unique_id_to_metadata.insert(unique_type_id, metadata).is_some() {
            bug!("Type metadata for unique id '{}' is already in the TypeMap!",
                 self.get_unique_type_id_as_string(unique_type_id));
        }
    }

    fn find_metadata_for_type(&self, type_: Ty<'tcx>) -> Option<&'ll DIType> {
        self.type_to_metadata.get(&type_).cloned()
    }

    fn find_metadata_for_unique_id(&self, unique_type_id: UniqueTypeId) -> Option<&'ll DIType> {
        self.unique_id_to_metadata.get(&unique_type_id).cloned()
    }

    // Get the string representation of a UniqueTypeId. This method will fail if
    // the id is unknown.
    fn get_unique_type_id_as_string(&self, unique_type_id: UniqueTypeId) -> &str {
        let UniqueTypeId(interner_key) = unique_type_id;
        self.unique_id_interner.get(interner_key)
    }

    // Get the UniqueTypeId for the given type. If the UniqueTypeId for the given
    // type has been requested before, this is just a table lookup. Otherwise an
    // ID will be generated and stored for later lookup.
    fn get_unique_type_id_of_type<'a>(&mut self, cx: &CodegenCx<'a, 'tcx>,
                                      type_: Ty<'tcx>) -> UniqueTypeId {
        // Let's see if we already have something in the cache
        if let Some(unique_type_id) = self.type_to_unique_id.get(&type_).cloned() {
            return unique_type_id;
        }
        // if not, generate one

        // The hasher we are using to generate the UniqueTypeId. We want
        // something that provides more than the 64 bits of the DefaultHasher.
        let mut hasher = StableHasher::<Fingerprint>::new();
        let mut hcx = cx.tcx.create_stable_hashing_context();
        let type_ = cx.tcx.erase_regions(&type_);
        hcx.while_hashing_spans(false, |hcx| {
            hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
                type_.hash_stable(hcx, &mut hasher);
            });
        });
        let unique_type_id = hasher.finish().to_hex();

        let key = self.unique_id_interner.intern(&unique_type_id);
        self.type_to_unique_id.insert(type_, UniqueTypeId(key));

        return UniqueTypeId(key);
    }

    // Get the UniqueTypeId for an enum variant. Enum variants are not really
    // types of their own, so they need special handling. We still need a
    // UniqueTypeId for them, since to debuginfo they *are* real types.
    fn get_unique_type_id_of_enum_variant<'a>(&mut self,
                                              cx: &CodegenCx<'a, 'tcx>,
                                              enum_type: Ty<'tcx>,
                                              variant_name: &str)
                                              -> UniqueTypeId {
        let enum_type_id = self.get_unique_type_id_of_type(cx, enum_type);
        let enum_variant_type_id = format!("{}::{}",
                                           self.get_unique_type_id_as_string(enum_type_id),
                                           variant_name);
        let interner_key = self.unique_id_interner.intern(&enum_variant_type_id);
        UniqueTypeId(interner_key)
    }

    // Get the unique type id string for an enum variant part.
    // Variant parts are not types and shouldn't really have their own id,
    // but it makes set_members_of_composite_type() simpler.
    fn get_unique_type_id_str_of_enum_variant_part(&mut self, enum_type_id: UniqueTypeId) -> &str {
        let variant_part_type_id = format!("{}_variant_part",
                                           self.get_unique_type_id_as_string(enum_type_id));
        let interner_key = self.unique_id_interner.intern(&variant_part_type_id);
        self.unique_id_interner.get(interner_key)
    }
}

// A description of some recursive type. It can either be already finished (as
// with FinalMetadata) or it is not yet finished, but contains all information
// needed to generate the missing parts of the description. See the
// documentation section on Recursive Types at the top of this file for more
// information.
enum RecursiveTypeDescription<'ll, 'tcx> {
    UnfinishedMetadata {
        unfinished_type: Ty<'tcx>,
        unique_type_id: UniqueTypeId,
        metadata_stub: &'ll DICompositeType,
        member_holding_stub: &'ll DICompositeType,
        member_description_factory: MemberDescriptionFactory<'ll, 'tcx>,
    },
    FinalMetadata(&'ll DICompositeType)
}

fn create_and_register_recursive_type_forward_declaration(
    cx: &CodegenCx<'ll, 'tcx>,
    unfinished_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId,
    metadata_stub: &'ll DICompositeType,
    member_holding_stub: &'ll DICompositeType,
    member_description_factory: MemberDescriptionFactory<'ll, 'tcx>,
) -> RecursiveTypeDescription<'ll, 'tcx> {

    // Insert the stub into the TypeMap in order to allow for recursive references
    let mut type_map = debug_context(cx).type_map.borrow_mut();
    type_map.register_unique_id_with_metadata(unique_type_id, metadata_stub);
    type_map.register_type_with_metadata(unfinished_type, metadata_stub);

    UnfinishedMetadata {
        unfinished_type,
        unique_type_id,
        metadata_stub,
        member_holding_stub,
        member_description_factory,
    }
}

impl RecursiveTypeDescription<'ll, 'tcx> {
    // Finishes up the description of the type in question (mostly by providing
    // descriptions of the fields of the given type) and returns the final type
    // metadata.
    fn finalize(&self, cx: &CodegenCx<'ll, 'tcx>) -> MetadataCreationResult<'ll> {
        match *self {
            FinalMetadata(metadata) => MetadataCreationResult::new(metadata, false),
            UnfinishedMetadata {
                unfinished_type,
                unique_type_id,
                metadata_stub,
                member_holding_stub,
                ref member_description_factory,
            } => {
                // Make sure that we have a forward declaration of the type in
                // the TypeMap so that recursive references are possible. This
                // will always be the case if the RecursiveTypeDescription has
                // been properly created through the
                // create_and_register_recursive_type_forward_declaration()
                // function.
                {
                    let type_map = debug_context(cx).type_map.borrow();
                    if type_map.find_metadata_for_unique_id(unique_type_id).is_none() ||
                       type_map.find_metadata_for_type(unfinished_type).is_none() {
                        bug!("Forward declaration of potentially recursive type \
                              '{:?}' was not found in TypeMap!",
                             unfinished_type);
                    }
                }

                // ... then create the member descriptions ...
                let member_descriptions =
                    member_description_factory.create_member_descriptions(cx);

                // ... and attach them to the stub to complete it.
                set_members_of_composite_type(cx,
                                              unfinished_type,
                                              member_holding_stub,
                                              member_descriptions);
                return MetadataCreationResult::new(metadata_stub, true);
            }
        }
    }
}

// Returns from the enclosing function if the type metadata with the given
// unique id can be found in the type map
macro_rules! return_if_metadata_created_in_meantime {
    ($cx: expr, $unique_type_id: expr) => (
        if let Some(metadata) = debug_context($cx).type_map
            .borrow()
            .find_metadata_for_unique_id($unique_type_id)
        {
            return MetadataCreationResult::new(metadata, true);
        }
    )
}

fn fixed_vec_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId,
    array_or_slice_type: Ty<'tcx>,
    element_type: Ty<'tcx>,
    span: Span,
) -> MetadataCreationResult<'ll> {
    let element_type_metadata = type_metadata(cx, element_type, span);

    return_if_metadata_created_in_meantime!(cx, unique_type_id);

    let (size, align) = cx.size_and_align_of(array_or_slice_type);

    let upper_bound = match array_or_slice_type.sty {
        ty::Array(_, len) => len.unwrap_usize(cx.tcx) as c_longlong,
        _ => -1
    };

    let subrange = unsafe {
        Some(llvm::LLVMRustDIBuilderGetOrCreateSubrange(DIB(cx), 0, upper_bound))
    };

    let subscripts = create_DIArray(DIB(cx), &[subrange]);
    let metadata = unsafe {
        llvm::LLVMRustDIBuilderCreateArrayType(
            DIB(cx),
            size.bits(),
            align.bits() as u32,
            element_type_metadata,
            subscripts)
    };

    return MetadataCreationResult::new(metadata, false);
}

fn vec_slice_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    slice_ptr_type: Ty<'tcx>,
    element_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId,
    span: Span,
) -> MetadataCreationResult<'ll> {
    let data_ptr_type = cx.tcx.mk_imm_ptr(element_type);

    let data_ptr_metadata = type_metadata(cx, data_ptr_type, span);

    return_if_metadata_created_in_meantime!(cx, unique_type_id);

    let slice_type_name = compute_debuginfo_type_name(cx.tcx, slice_ptr_type, true);

    let (pointer_size, pointer_align) = cx.size_and_align_of(data_ptr_type);
    let (usize_size, usize_align) = cx.size_and_align_of(cx.tcx.types.usize);

    let member_descriptions = vec![
        MemberDescription {
            name: "data_ptr".to_owned(),
            type_metadata: data_ptr_metadata,
            offset: Size::ZERO,
            size: pointer_size,
            align: pointer_align,
            flags: DIFlags::FlagZero,
            discriminant: None,
        },
        MemberDescription {
            name: "length".to_owned(),
            type_metadata: type_metadata(cx, cx.tcx.types.usize, span),
            offset: pointer_size,
            size: usize_size,
            align: usize_align,
            flags: DIFlags::FlagZero,
            discriminant: None,
        },
    ];

    let file_metadata = unknown_file_metadata(cx);

    let metadata = composite_type_metadata(cx,
                                           slice_ptr_type,
                                           &slice_type_name[..],
                                           unique_type_id,
                                           member_descriptions,
                                           NO_SCOPE_METADATA,
                                           file_metadata,
                                           span);
    MetadataCreationResult::new(metadata, false)
}

fn subroutine_type_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId,
    signature: ty::PolyFnSig<'tcx>,
    span: Span,
) -> MetadataCreationResult<'ll> {
    let signature = cx.tcx.normalize_erasing_late_bound_regions(
        ty::ParamEnv::reveal_all(),
        &signature,
    );

    let signature_metadata: Vec<_> = iter::once(
        // return type
        match signature.output().sty {
            ty::Tuple(ref tys) if tys.is_empty() => None,
            _ => Some(type_metadata(cx, signature.output(), span))
        }
    ).chain(
        // regular arguments
        signature.inputs().iter().map(|argument_type| {
            Some(type_metadata(cx, argument_type, span))
        })
    ).collect();

    return_if_metadata_created_in_meantime!(cx, unique_type_id);

    return MetadataCreationResult::new(
        unsafe {
            llvm::LLVMRustDIBuilderCreateSubroutineType(
                DIB(cx),
                unknown_file_metadata(cx),
                create_DIArray(DIB(cx), &signature_metadata[..]))
        },
        false);
}

// FIXME(1563) This is all a bit of a hack because 'trait pointer' is an ill-
// defined concept. For the case of an actual trait pointer (i.e., Box<Trait>,
// &Trait), trait_object_type should be the whole thing (e.g, Box<Trait>) and
// trait_type should be the actual trait (e.g., Trait). Where the trait is part
// of a DST struct, there is no trait_object_type and the results of this
// function will be a little bit weird.
fn trait_pointer_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    trait_type: Ty<'tcx>,
    trait_object_type: Option<Ty<'tcx>>,
    unique_type_id: UniqueTypeId,
) -> &'ll DIType {
    // The implementation provided here is a stub. It makes sure that the trait
    // type is assigned the correct name, size, namespace, and source location.
    // But it does not describe the trait's methods.

    let containing_scope = match trait_type.sty {
        ty::Dynamic(ref data, ..) =>
            data.principal_def_id().map(|did| get_namespace_for_item(cx, did)),
        _ => {
            bug!("debuginfo: Unexpected trait-object type in \
                  trait_pointer_metadata(): {:?}",
                 trait_type);
        }
    };

    let trait_object_type = trait_object_type.unwrap_or(trait_type);
    let trait_type_name =
        compute_debuginfo_type_name(cx.tcx, trait_object_type, false);

    let file_metadata = unknown_file_metadata(cx);

    let layout = cx.layout_of(cx.tcx.mk_mut_ptr(trait_type));

    assert_eq!(abi::FAT_PTR_ADDR, 0);
    assert_eq!(abi::FAT_PTR_EXTRA, 1);

    let data_ptr_field = layout.field(cx, 0);
    let vtable_field = layout.field(cx, 1);
    let member_descriptions = vec![
        MemberDescription {
            name: "pointer".to_owned(),
            type_metadata: type_metadata(cx,
                cx.tcx.mk_mut_ptr(cx.tcx.types.u8),
                syntax_pos::DUMMY_SP),
            offset: layout.fields.offset(0),
            size: data_ptr_field.size,
            align: data_ptr_field.align.abi,
            flags: DIFlags::FlagArtificial,
            discriminant: None,
        },
        MemberDescription {
            name: "vtable".to_owned(),
            type_metadata: type_metadata(cx, vtable_field.ty, syntax_pos::DUMMY_SP),
            offset: layout.fields.offset(1),
            size: vtable_field.size,
            align: vtable_field.align.abi,
            flags: DIFlags::FlagArtificial,
            discriminant: None,
        },
    ];

    composite_type_metadata(cx,
                            trait_object_type,
                            &trait_type_name[..],
                            unique_type_id,
                            member_descriptions,
                            containing_scope,
                            file_metadata,
                            syntax_pos::DUMMY_SP)
}

pub fn type_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    t: Ty<'tcx>,
    usage_site_span: Span,
) -> &'ll DIType {
    // Get the unique type id of this type.
    let unique_type_id = {
        let mut type_map = debug_context(cx).type_map.borrow_mut();
        // First, try to find the type in TypeMap. If we have seen it before, we
        // can exit early here.
        match type_map.find_metadata_for_type(t) {
            Some(metadata) => {
                return metadata;
            },
            None => {
                // The Ty is not in the TypeMap but maybe we have already seen
                // an equivalent type (e.g., only differing in region arguments).
                // In order to find out, generate the unique type id and look
                // that up.
                let unique_type_id = type_map.get_unique_type_id_of_type(cx, t);
                match type_map.find_metadata_for_unique_id(unique_type_id) {
                    Some(metadata) => {
                        // There is already an equivalent type in the TypeMap.
                        // Register this Ty as an alias in the cache and
                        // return the cached metadata.
                        type_map.register_type_with_metadata(t, metadata);
                        return metadata;
                    },
                    None => {
                        // There really is no type metadata for this type, so
                        // proceed by creating it.
                        unique_type_id
                    }
                }
            }
        }
    };

    debug!("type_metadata: {:?}", t);

    let ptr_metadata = |ty: Ty<'tcx>| {
        match ty.sty {
            ty::Slice(typ) => {
                Ok(vec_slice_metadata(cx, t, typ, unique_type_id, usage_site_span))
            }
            ty::Str => {
                Ok(vec_slice_metadata(cx, t, cx.tcx.types.u8, unique_type_id, usage_site_span))
            }
            ty::Dynamic(..) => {
                Ok(MetadataCreationResult::new(
                    trait_pointer_metadata(cx, ty, Some(t), unique_type_id),
                    false))
            }
            _ => {
                let pointee_metadata = type_metadata(cx, ty, usage_site_span);

                if let Some(metadata) = debug_context(cx).type_map
                    .borrow()
                    .find_metadata_for_unique_id(unique_type_id)
                {
                    return Err(metadata);
                }

                Ok(MetadataCreationResult::new(pointer_type_metadata(cx, t, pointee_metadata),
                   false))
            }
        }
    };

    let MetadataCreationResult { metadata, already_stored_in_typemap } = match t.sty {
        ty::Never    |
        ty::Bool     |
        ty::Char     |
        ty::Int(_)   |
        ty::Uint(_)  |
        ty::Float(_) => {
            MetadataCreationResult::new(basic_type_metadata(cx, t), false)
        }
        ty::Tuple(ref elements) if elements.is_empty() => {
            MetadataCreationResult::new(basic_type_metadata(cx, t), false)
        }
        ty::Array(typ, _) |
        ty::Slice(typ) => {
            fixed_vec_metadata(cx, unique_type_id, t, typ, usage_site_span)
        }
        ty::Str => {
            fixed_vec_metadata(cx, unique_type_id, t, cx.tcx.types.i8, usage_site_span)
        }
        ty::Dynamic(..) => {
            MetadataCreationResult::new(
                trait_pointer_metadata(cx, t, None, unique_type_id),
                false)
        }
        ty::Foreign(..) => {
            MetadataCreationResult::new(
            foreign_type_metadata(cx, t, unique_type_id),
            false)
        }
        ty::RawPtr(ty::TypeAndMut{ty, ..}) |
        ty::Ref(_, ty, _) => {
            match ptr_metadata(ty) {
                Ok(res) => res,
                Err(metadata) => return metadata,
            }
        }
        ty::Adt(def, _) if def.is_box() => {
            match ptr_metadata(t.boxed_ty()) {
                Ok(res) => res,
                Err(metadata) => return metadata,
            }
        }
        ty::FnDef(..) | ty::FnPtr(_) => {

            if let Some(metadata) = debug_context(cx).type_map
               .borrow()
               .find_metadata_for_unique_id(unique_type_id)
            {
                return metadata;
            }

            // It's possible to create a self-referential
            // type in Rust by using 'impl trait':
            //
            // fn foo() -> impl Copy { foo }
            //
            // See TypeMap::remove_type for more detals
            // about the workaround

            let temp_type = {
                unsafe {
                    // The choice of type here is pretty arbitrary -
                    // anything reading the debuginfo for a recursive
                    // type is going to see *somthing* weird - the only
                    // question is what exactly it will see
                    let (size, align) = cx.size_and_align_of(t);
                    llvm::LLVMRustDIBuilderCreateBasicType(
                        DIB(cx),
                        SmallCStr::new("<recur_type>").as_ptr(),
                        size.bits(),
                        align.bits() as u32,
                        DW_ATE_unsigned)
                }
            };

            let type_map = &debug_context(cx).type_map;
            type_map.borrow_mut().register_type_with_metadata(t, temp_type);

            let fn_metadata = subroutine_type_metadata(cx,
                                                       unique_type_id,
                                                       t.fn_sig(cx.tcx),
                                                       usage_site_span).metadata;

            type_map.borrow_mut().remove_type(t);


            // This is actually a function pointer, so wrap it in pointer DI
            MetadataCreationResult::new(pointer_type_metadata(cx, t, fn_metadata), false)

        }
        ty::Closure(def_id, substs) => {
            let upvar_tys : Vec<_> = substs.upvar_tys(def_id, cx.tcx).collect();
            prepare_tuple_metadata(cx,
                                   t,
                                   &upvar_tys,
                                   unique_type_id,
                                   usage_site_span).finalize(cx)
        }
        ty::Generator(def_id, substs,  _) => {
            let upvar_tys : Vec<_> = substs.prefix_tys(def_id, cx.tcx).map(|t| {
                cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), t)
            }).collect();
            prepare_enum_metadata(cx,
                                  t,
                                  def_id,
                                  unique_type_id,
                                  usage_site_span,
                                  upvar_tys).finalize(cx)
        }
        ty::Adt(def, ..) => match def.adt_kind() {
            AdtKind::Struct => {
                prepare_struct_metadata(cx,
                                        t,
                                        unique_type_id,
                                        usage_site_span).finalize(cx)
            }
            AdtKind::Union => {
                prepare_union_metadata(cx,
                                       t,
                                       unique_type_id,
                                       usage_site_span).finalize(cx)
            }
            AdtKind::Enum => {
                prepare_enum_metadata(cx,
                                      t,
                                      def.did,
                                      unique_type_id,
                                      usage_site_span,
                                      vec![]).finalize(cx)
            }
        },
        ty::Tuple(ref elements) => {
            let tys: Vec<_> = elements.iter().map(|k| k.expect_ty()).collect();
            prepare_tuple_metadata(cx,
                                   t,
                                   &tys,
                                   unique_type_id,
                                   usage_site_span).finalize(cx)
        }
        _ => {
            bug!("debuginfo: unexpected type in type_metadata: {:?}", t)
        }
    };

    {
        let mut type_map = debug_context(cx).type_map.borrow_mut();

        if already_stored_in_typemap {
            // Also make sure that we already have a TypeMap entry for the unique type id.
            let metadata_for_uid = match type_map.find_metadata_for_unique_id(unique_type_id) {
                Some(metadata) => metadata,
                None => {
                    span_bug!(usage_site_span,
                              "Expected type metadata for unique \
                               type id '{}' to already be in \
                               the debuginfo::TypeMap but it \
                               was not. (Ty = {})",
                              type_map.get_unique_type_id_as_string(unique_type_id),
                              t);
                }
            };

            match type_map.find_metadata_for_type(t) {
                Some(metadata) => {
                    if metadata != metadata_for_uid {
                        span_bug!(usage_site_span,
                                  "Mismatch between Ty and \
                                   UniqueTypeId maps in \
                                   debuginfo::TypeMap. \
                                   UniqueTypeId={}, Ty={}",
                                  type_map.get_unique_type_id_as_string(unique_type_id),
                                  t);
                    }
                }
                None => {
                    type_map.register_type_with_metadata(t, metadata);
                }
            }
        } else {
            type_map.register_type_with_metadata(t, metadata);
            type_map.register_unique_id_with_metadata(unique_type_id, metadata);
        }
    }

    metadata
}

pub fn file_metadata(cx: &CodegenCx<'ll, '_>,
                     file_name: &FileName,
                     defining_crate: CrateNum) -> &'ll DIFile {
    debug!("file_metadata: file_name: {}, defining_crate: {}",
           file_name,
           defining_crate);

    let file_name = Some(file_name.to_string());
    let directory = if defining_crate == LOCAL_CRATE {
        Some(cx.sess().working_dir.0.to_string_lossy().to_string())
    } else {
        // If the path comes from an upstream crate we assume it has been made
        // independent of the compiler's working directory one way or another.
        None
    };
    file_metadata_raw(cx, file_name, directory)
}

pub fn unknown_file_metadata(cx: &CodegenCx<'ll, '_>) -> &'ll DIFile {
    file_metadata_raw(cx, None, None)
}

fn file_metadata_raw(cx: &CodegenCx<'ll, '_>,
                     file_name: Option<String>,
                     directory: Option<String>)
                     -> &'ll DIFile {
    let key = (file_name, directory);

    match debug_context(cx).created_files.borrow_mut().entry(key) {
        Entry::Occupied(o) => return o.get(),
        Entry::Vacant(v) => {
            let (file_name, directory) = v.key();
            debug!("file_metadata: file_name: {:?}, directory: {:?}", file_name, directory);

            let file_name = SmallCStr::new(
                if let Some(file_name) = file_name { &file_name } else { "<unknown>" });
            let directory = SmallCStr::new(
                if let Some(directory) = directory { &directory } else { "" });

            let file_metadata = unsafe {
                llvm::LLVMRustDIBuilderCreateFile(DIB(cx),
                                                  file_name.as_ptr(),
                                                  directory.as_ptr())
            };

            v.insert(file_metadata);
            file_metadata
        }
    }
}

fn basic_type_metadata(cx: &CodegenCx<'ll, 'tcx>, t: Ty<'tcx>) -> &'ll DIType {
    debug!("basic_type_metadata: {:?}", t);

    let (name, encoding) = match t.sty {
        ty::Never => ("!", DW_ATE_unsigned),
        ty::Tuple(ref elements) if elements.is_empty() =>
            ("()", DW_ATE_unsigned),
        ty::Bool => ("bool", DW_ATE_boolean),
        ty::Char => ("char", DW_ATE_unsigned_char),
        ty::Int(int_ty) => {
            (int_ty.ty_to_string(), DW_ATE_signed)
        },
        ty::Uint(uint_ty) => {
            (uint_ty.ty_to_string(), DW_ATE_unsigned)
        },
        ty::Float(float_ty) => {
            (float_ty.ty_to_string(), DW_ATE_float)
        },
        _ => bug!("debuginfo::basic_type_metadata - t is invalid type")
    };

    let (size, align) = cx.size_and_align_of(t);
    let name = SmallCStr::new(name);
    let ty_metadata = unsafe {
        llvm::LLVMRustDIBuilderCreateBasicType(
            DIB(cx),
            name.as_ptr(),
            size.bits(),
            align.bits() as u32,
            encoding)
    };

    return ty_metadata;
}

fn foreign_type_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    t: Ty<'tcx>,
    unique_type_id: UniqueTypeId,
) -> &'ll DIType {
    debug!("foreign_type_metadata: {:?}", t);

    let name = compute_debuginfo_type_name(cx.tcx, t, false);
    create_struct_stub(cx, t, &name, unique_type_id, NO_SCOPE_METADATA)
}

fn pointer_type_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    pointer_type: Ty<'tcx>,
    pointee_type_metadata: &'ll DIType,
) -> &'ll DIType {
    let (pointer_size, pointer_align) = cx.size_and_align_of(pointer_type);
    let name = compute_debuginfo_type_name(cx.tcx, pointer_type, false);
    let name = SmallCStr::new(&name);
    unsafe {
        llvm::LLVMRustDIBuilderCreatePointerType(
            DIB(cx),
            pointee_type_metadata,
            pointer_size.bits(),
            pointer_align.bits() as u32,
            name.as_ptr())
    }
}

pub fn compile_unit_metadata(
    tcx: TyCtxt<'_>,
    codegen_unit_name: &str,
    debug_context: &CrateDebugContext<'ll, '_>,
) -> &'ll DIDescriptor {
    let mut name_in_debuginfo = match tcx.sess.local_crate_source_file {
        Some(ref path) => path.clone(),
        None => PathBuf::from(&*tcx.crate_name(LOCAL_CRATE).as_str()),
    };

    // The OSX linker has an idiosyncrasy where it will ignore some debuginfo
    // if multiple object files with the same DW_AT_name are linked together.
    // As a workaround we generate unique names for each object file. Those do
    // not correspond to an actual source file but that should be harmless.
    if tcx.sess.target.target.options.is_like_osx {
        name_in_debuginfo.push("@");
        name_in_debuginfo.push(codegen_unit_name);
    }

    debug!("compile_unit_metadata: {:?}", name_in_debuginfo);
    // FIXME(#41252) Remove "clang LLVM" if we can get GDB and LLVM to play nice.
    let producer = format!("clang LLVM (rustc version {})",
                           (option_env!("CFG_VERSION")).expect("CFG_VERSION"));

    let name_in_debuginfo = name_in_debuginfo.to_string_lossy();
    let name_in_debuginfo = SmallCStr::new(&name_in_debuginfo);
    let work_dir = SmallCStr::new(&tcx.sess.working_dir.0.to_string_lossy());
    let producer = CString::new(producer).unwrap();
    let flags = "\0";
    let split_name = "\0";

    // FIXME(#60020):
    //
    //    This should actually be
    //
    //    ```
    //      let kind = DebugEmissionKind::from_generic(tcx.sess.opts.debuginfo);
    //    ```
    //
    //    that is, we should set LLVM's emission kind to `LineTablesOnly` if
    //    we are compiling with "limited" debuginfo. However, some of the
    //    existing tools relied on slightly more debuginfo being generated than
    //    would be the case with `LineTablesOnly`, and we did not want to break
    //    these tools in a "drive-by fix", without a good idea or plan about
    //    what limited debuginfo should exactly look like. So for now we keep
    //    the emission kind as `FullDebug`.
    //
    //    See https://github.com/rust-lang/rust/issues/60020 for details.
    let kind = DebugEmissionKind::FullDebug;
    assert!(tcx.sess.opts.debuginfo != DebugInfo::None);

    unsafe {
        let file_metadata = llvm::LLVMRustDIBuilderCreateFile(
            debug_context.builder, name_in_debuginfo.as_ptr(), work_dir.as_ptr());

        let unit_metadata = llvm::LLVMRustDIBuilderCreateCompileUnit(
            debug_context.builder,
            DW_LANG_RUST,
            file_metadata,
            producer.as_ptr(),
            tcx.sess.opts.optimize != config::OptLevel::No,
            flags.as_ptr() as *const _,
            0,
            split_name.as_ptr() as *const _,
            kind);

        if tcx.sess.opts.debugging_opts.profile {
            let cu_desc_metadata = llvm::LLVMRustMetadataAsValue(debug_context.llcontext,
                                                                 unit_metadata);

            let gcov_cu_info = [
                path_to_mdstring(debug_context.llcontext,
                                 &tcx.output_filenames(LOCAL_CRATE).with_extension("gcno")),
                path_to_mdstring(debug_context.llcontext,
                                 &tcx.output_filenames(LOCAL_CRATE).with_extension("gcda")),
                cu_desc_metadata,
            ];
            let gcov_metadata = llvm::LLVMMDNodeInContext(debug_context.llcontext,
                                                          gcov_cu_info.as_ptr(),
                                                          gcov_cu_info.len() as c_uint);

            let llvm_gcov_ident = const_cstr!("llvm.gcov");
            llvm::LLVMAddNamedMetadataOperand(debug_context.llmod,
                                              llvm_gcov_ident.as_ptr(),
                                              gcov_metadata);
        }

        return unit_metadata;
    };

    fn path_to_mdstring(llcx: &'ll llvm::Context, path: &Path) -> &'ll Value {
        let path_str = path_to_c_string(path);
        unsafe {
            llvm::LLVMMDStringInContext(llcx,
                                        path_str.as_ptr(),
                                        path_str.as_bytes().len() as c_uint)
        }
    }
}

struct MetadataCreationResult<'ll> {
    metadata: &'ll DIType,
    already_stored_in_typemap: bool
}

impl MetadataCreationResult<'ll> {
    fn new(metadata: &'ll DIType, already_stored_in_typemap: bool) -> Self {
        MetadataCreationResult {
            metadata,
            already_stored_in_typemap,
        }
    }
}

// Description of a type member, which can either be a regular field (as in
// structs or tuples) or an enum variant.
#[derive(Debug)]
struct MemberDescription<'ll> {
    name: String,
    type_metadata: &'ll DIType,
    offset: Size,
    size: Size,
    align: Align,
    flags: DIFlags,
    discriminant: Option<u64>,
}

impl<'ll> MemberDescription<'ll> {
    fn into_metadata(self,
                     cx: &CodegenCx<'ll, '_>,
                     composite_type_metadata: &'ll DIScope) -> &'ll DIType {
        let member_name = CString::new(self.name).unwrap();
        unsafe {
            llvm::LLVMRustDIBuilderCreateVariantMemberType(
                DIB(cx),
                composite_type_metadata,
                member_name.as_ptr(),
                unknown_file_metadata(cx),
                UNKNOWN_LINE_NUMBER,
                self.size.bits(),
                self.align.bits() as u32,
                self.offset.bits(),
                match self.discriminant {
                    None => None,
                    Some(value) => Some(cx.const_u64(value)),
                },
                self.flags,
                self.type_metadata)
        }
    }
}

// A factory for MemberDescriptions. It produces a list of member descriptions
// for some record-like type. MemberDescriptionFactories are used to defer the
// creation of type member descriptions in order to break cycles arising from
// recursive type definitions.
enum MemberDescriptionFactory<'ll, 'tcx> {
    StructMDF(StructMemberDescriptionFactory<'tcx>),
    TupleMDF(TupleMemberDescriptionFactory<'tcx>),
    EnumMDF(EnumMemberDescriptionFactory<'ll, 'tcx>),
    UnionMDF(UnionMemberDescriptionFactory<'tcx>),
    VariantMDF(VariantMemberDescriptionFactory<'ll, 'tcx>)
}

impl MemberDescriptionFactory<'ll, 'tcx> {
    fn create_member_descriptions(&self, cx: &CodegenCx<'ll, 'tcx>)
                                  -> Vec<MemberDescription<'ll>> {
        match *self {
            StructMDF(ref this) => {
                this.create_member_descriptions(cx)
            }
            TupleMDF(ref this) => {
                this.create_member_descriptions(cx)
            }
            EnumMDF(ref this) => {
                this.create_member_descriptions(cx)
            }
            UnionMDF(ref this) => {
                this.create_member_descriptions(cx)
            }
            VariantMDF(ref this) => {
                this.create_member_descriptions(cx)
            }
        }
    }
}

//=-----------------------------------------------------------------------------
// Structs
//=-----------------------------------------------------------------------------

// Creates MemberDescriptions for the fields of a struct
struct StructMemberDescriptionFactory<'tcx> {
    ty: Ty<'tcx>,
    variant: &'tcx ty::VariantDef,
    span: Span,
}

impl<'tcx> StructMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions(&self, cx: &CodegenCx<'ll, 'tcx>)
                                  -> Vec<MemberDescription<'ll>> {
        let layout = cx.layout_of(self.ty);
        self.variant.fields.iter().enumerate().map(|(i, f)| {
            let name = if self.variant.ctor_kind == CtorKind::Fn {
                format!("__{}", i)
            } else {
                f.ident.to_string()
            };
            let field = layout.field(cx, i);
            MemberDescription {
                name,
                type_metadata: type_metadata(cx, field.ty, self.span),
                offset: layout.fields.offset(i),
                size: field.size,
                align: field.align.abi,
                flags: DIFlags::FlagZero,
                discriminant: None,
            }
        }).collect()
    }
}


fn prepare_struct_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    struct_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId,
    span: Span,
) -> RecursiveTypeDescription<'ll, 'tcx> {
    let struct_name = compute_debuginfo_type_name(cx.tcx, struct_type, false);

    let (struct_def_id, variant) = match struct_type.sty {
        ty::Adt(def, _) => (def.did, def.non_enum_variant()),
        _ => bug!("prepare_struct_metadata on a non-ADT")
    };

    let containing_scope = get_namespace_for_item(cx, struct_def_id);

    let struct_metadata_stub = create_struct_stub(cx,
                                                  struct_type,
                                                  &struct_name,
                                                  unique_type_id,
                                                  Some(containing_scope));

    create_and_register_recursive_type_forward_declaration(
        cx,
        struct_type,
        unique_type_id,
        struct_metadata_stub,
        struct_metadata_stub,
        StructMDF(StructMemberDescriptionFactory {
            ty: struct_type,
            variant,
            span,
        })
    )
}

//=-----------------------------------------------------------------------------
// Tuples
//=-----------------------------------------------------------------------------

// Creates MemberDescriptions for the fields of a tuple
struct TupleMemberDescriptionFactory<'tcx> {
    ty: Ty<'tcx>,
    component_types: Vec<Ty<'tcx>>,
    span: Span,
}

impl<'tcx> TupleMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions(&self, cx: &CodegenCx<'ll, 'tcx>)
                                  -> Vec<MemberDescription<'ll>> {
        let layout = cx.layout_of(self.ty);
        self.component_types.iter().enumerate().map(|(i, &component_type)| {
            let (size, align) = cx.size_and_align_of(component_type);
            MemberDescription {
                name: format!("__{}", i),
                type_metadata: type_metadata(cx, component_type, self.span),
                offset: layout.fields.offset(i),
                size,
                align,
                flags: DIFlags::FlagZero,
                discriminant: None,
            }
        }).collect()
    }
}

fn prepare_tuple_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    tuple_type: Ty<'tcx>,
    component_types: &[Ty<'tcx>],
    unique_type_id: UniqueTypeId,
    span: Span,
) -> RecursiveTypeDescription<'ll, 'tcx> {
    let tuple_name = compute_debuginfo_type_name(cx.tcx, tuple_type, false);

    let struct_stub = create_struct_stub(cx,
                                         tuple_type,
                                         &tuple_name[..],
                                         unique_type_id,
                                         NO_SCOPE_METADATA);

    create_and_register_recursive_type_forward_declaration(
        cx,
        tuple_type,
        unique_type_id,
        struct_stub,
        struct_stub,
        TupleMDF(TupleMemberDescriptionFactory {
            ty: tuple_type,
            component_types: component_types.to_vec(),
            span,
        })
    )
}

//=-----------------------------------------------------------------------------
// Unions
//=-----------------------------------------------------------------------------

struct UnionMemberDescriptionFactory<'tcx> {
    layout: TyLayout<'tcx>,
    variant: &'tcx ty::VariantDef,
    span: Span,
}

impl<'tcx> UnionMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions(&self, cx: &CodegenCx<'ll, 'tcx>)
                                  -> Vec<MemberDescription<'ll>> {
        self.variant.fields.iter().enumerate().map(|(i, f)| {
            let field = self.layout.field(cx, i);
            MemberDescription {
                name: f.ident.to_string(),
                type_metadata: type_metadata(cx, field.ty, self.span),
                offset: Size::ZERO,
                size: field.size,
                align: field.align.abi,
                flags: DIFlags::FlagZero,
                discriminant: None,
            }
        }).collect()
    }
}

fn prepare_union_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    union_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId,
    span: Span,
) -> RecursiveTypeDescription<'ll, 'tcx> {
    let union_name = compute_debuginfo_type_name(cx.tcx, union_type, false);

    let (union_def_id, variant) = match union_type.sty {
        ty::Adt(def, _) => (def.did, def.non_enum_variant()),
        _ => bug!("prepare_union_metadata on a non-ADT")
    };

    let containing_scope = get_namespace_for_item(cx, union_def_id);

    let union_metadata_stub = create_union_stub(cx,
                                                union_type,
                                                &union_name,
                                                unique_type_id,
                                                containing_scope);

    create_and_register_recursive_type_forward_declaration(
        cx,
        union_type,
        unique_type_id,
        union_metadata_stub,
        union_metadata_stub,
        UnionMDF(UnionMemberDescriptionFactory {
            layout: cx.layout_of(union_type),
            variant,
            span,
        })
    )
}

//=-----------------------------------------------------------------------------
// Enums
//=-----------------------------------------------------------------------------

// DWARF variant support is only available starting in LLVM 8.
// Although the earlier enum debug info output did not work properly
// in all situations, it is better for the time being to continue to
// sometimes emit the old style rather than emit something completely
// useless when rust is compiled against LLVM 6 or older. LLVM 7
// contains an early version of the DWARF variant support, and will
// crash when handling the new debug info format. This function
// decides which representation will be emitted.
fn use_enum_fallback(cx: &CodegenCx<'_, '_>) -> bool {
    // On MSVC we have to use the fallback mode, because LLVM doesn't
    // lower variant parts to PDB.
    return cx.sess().target.target.options.is_like_msvc
        // LLVM version 7 did not release with an important bug fix;
        // but the required patch is in the LLVM 8.  Rust LLVM reports
        // 8 as well.
        || llvm_util::get_major_version() < 8;
}

// Describes the members of an enum value: An enum is described as a union of
// structs in DWARF. This MemberDescriptionFactory provides the description for
// the members of this union; so for every variant of the given enum, this
// factory will produce one MemberDescription (all with no name and a fixed
// offset of zero bytes).
struct EnumMemberDescriptionFactory<'ll, 'tcx> {
    enum_type: Ty<'tcx>,
    layout: TyLayout<'tcx>,
    discriminant_type_metadata: Option<&'ll DIType>,
    containing_scope: &'ll DIScope,
    span: Span,
}

impl EnumMemberDescriptionFactory<'ll, 'tcx> {
    fn create_member_descriptions(&self, cx: &CodegenCx<'ll, 'tcx>)
                                  -> Vec<MemberDescription<'ll>> {
        let variant_info_for = |index: VariantIdx| {
            match &self.enum_type.sty {
                ty::Adt(adt, _) => VariantInfo::Adt(&adt.variants[index]),
                ty::Generator(def_id, substs, _) => {
                    let generator_layout = cx.tcx.generator_layout(*def_id);
                    VariantInfo::Generator(*substs, generator_layout, index)
                }
                _ => bug!(),
            }
        };

        // This will always find the metadata in the type map.
        let fallback = use_enum_fallback(cx);
        let self_metadata = if fallback {
            self.containing_scope
        } else {
            type_metadata(cx, self.enum_type, self.span)
        };

        match self.layout.variants {
            layout::Variants::Single { index } => {
                if let ty::Adt(adt, _) = &self.enum_type.sty {
                    if adt.variants.is_empty() {
                        return vec![];
                    }
                }

                let variant_info = variant_info_for(index);
                let (variant_type_metadata, member_description_factory) =
                    describe_enum_variant(cx,
                                          self.layout,
                                          variant_info,
                                          NoDiscriminant,
                                          self_metadata,
                                          self.span);

                let member_descriptions =
                    member_description_factory.create_member_descriptions(cx);

                set_members_of_composite_type(cx,
                                              self.enum_type,
                                              variant_type_metadata,
                                              member_descriptions);
                vec![
                    MemberDescription {
                        name: if fallback {
                            String::new()
                        } else {
                            variant_info.variant_name()
                        },
                        type_metadata: variant_type_metadata,
                        offset: Size::ZERO,
                        size: self.layout.size,
                        align: self.layout.align.abi,
                        flags: DIFlags::FlagZero,
                        discriminant: None,
                    }
                ]
            }
            layout::Variants::Multiple {
                discr_kind: layout::DiscriminantKind::Tag,
                discr_index,
                ref variants,
                ..
            } => {
                let discriminant_info = if fallback {
                    RegularDiscriminant {
                        discr_field: Field::from(discr_index),
                        discr_type_metadata: self.discriminant_type_metadata.unwrap()
                    }
                } else {
                    // This doesn't matter in this case.
                    NoDiscriminant
                };
                variants.iter_enumerated().map(|(i, _)| {
                    let variant = self.layout.for_variant(cx, i);
                    let variant_info = variant_info_for(i);
                    let (variant_type_metadata, member_desc_factory) =
                        describe_enum_variant(cx,
                                              variant,
                                              variant_info,
                                              discriminant_info,
                                              self_metadata,
                                              self.span);

                    let member_descriptions = member_desc_factory
                        .create_member_descriptions(cx);

                    set_members_of_composite_type(cx,
                                                  self.enum_type,
                                                  variant_type_metadata,
                                                  member_descriptions);

                    MemberDescription {
                        name: if fallback {
                            String::new()
                        } else {
                            variant_info.variant_name()
                        },
                        type_metadata: variant_type_metadata,
                        offset: Size::ZERO,
                        size: self.layout.size,
                        align: self.layout.align.abi,
                        flags: DIFlags::FlagZero,
                        discriminant: Some(
                            self.layout.ty.discriminant_for_variant(cx.tcx, i).unwrap().val as u64
                        ),
                    }
                }).collect()
            }
            layout::Variants::Multiple {
                discr_kind: layout::DiscriminantKind::Niche {
                    ref niche_variants,
                    niche_start,
                    dataful_variant,
                },
                ref discr,
                ref variants,
                discr_index,
            } => {
                if fallback {
                    let variant = self.layout.for_variant(cx, dataful_variant);
                    // Create a description of the non-null variant
                    let (variant_type_metadata, member_description_factory) =
                        describe_enum_variant(cx,
                                              variant,
                                              variant_info_for(dataful_variant),
                                              OptimizedDiscriminant,
                                              self.containing_scope,
                                              self.span);

                    let variant_member_descriptions =
                        member_description_factory.create_member_descriptions(cx);

                    set_members_of_composite_type(cx,
                                                  self.enum_type,
                                                  variant_type_metadata,
                                                  variant_member_descriptions);

                    // Encode the information about the null variant in the union
                    // member's name.
                    let mut name = String::from("RUST$ENCODED$ENUM$");
                    // Right now it's not even going to work for `niche_start > 0`,
                    // and for multiple niche variants it only supports the first.
                    fn compute_field_path<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                                    name: &mut String,
                                                    layout: TyLayout<'tcx>,
                                                    offset: Size,
                                                    size: Size) {
                        for i in 0..layout.fields.count() {
                            let field_offset = layout.fields.offset(i);
                            if field_offset > offset {
                                continue;
                            }
                            let inner_offset = offset - field_offset;
                            let field = layout.field(cx, i);
                            if inner_offset + size <= field.size {
                                write!(name, "{}$", i).unwrap();
                                compute_field_path(cx, name, field, inner_offset, size);
                            }
                        }
                    }
                    compute_field_path(cx, &mut name,
                                       self.layout,
                                       self.layout.fields.offset(discr_index),
                                       self.layout.field(cx, discr_index).size);
                    variant_info_for(*niche_variants.start()).map_struct_name(|variant_name| {
                        name.push_str(variant_name);
                    });

                    // Create the (singleton) list of descriptions of union members.
                    vec![
                        MemberDescription {
                            name,
                            type_metadata: variant_type_metadata,
                            offset: Size::ZERO,
                            size: variant.size,
                            align: variant.align.abi,
                            flags: DIFlags::FlagZero,
                            discriminant: None,
                        }
                    ]
                } else {
                    variants.iter_enumerated().map(|(i, _)| {
                        let variant = self.layout.for_variant(cx, i);
                        let variant_info = variant_info_for(i);
                        let (variant_type_metadata, member_desc_factory) =
                            describe_enum_variant(cx,
                                                  variant,
                                                  variant_info,
                                                  OptimizedDiscriminant,
                                                  self_metadata,
                                                  self.span);

                        let member_descriptions = member_desc_factory
                            .create_member_descriptions(cx);

                        set_members_of_composite_type(cx,
                                                      self.enum_type,
                                                      variant_type_metadata,
                                                      member_descriptions);

                        let niche_value = if i == dataful_variant {
                            None
                        } else {
                            let value = (i.as_u32() as u128)
                                .wrapping_sub(niche_variants.start().as_u32() as u128)
                                .wrapping_add(niche_start);
                            let value = truncate(value, discr.value.size(cx));
                            // NOTE(eddyb) do *NOT* remove this assert, until
                            // we pass the full 128-bit value to LLVM, otherwise
                            // truncation will be silent and remain undetected.
                            assert_eq!(value as u64 as u128, value);
                            Some(value as u64)
                        };

                        MemberDescription {
                            name: variant_info.variant_name(),
                            type_metadata: variant_type_metadata,
                            offset: Size::ZERO,
                            size: self.layout.size,
                            align: self.layout.align.abi,
                            flags: DIFlags::FlagZero,
                            discriminant: niche_value,
                        }
                    }).collect()
                }
            }
        }
    }
}

// Creates MemberDescriptions for the fields of a single enum variant.
struct VariantMemberDescriptionFactory<'ll, 'tcx> {
    // Cloned from the layout::Struct describing the variant.
    offsets: Vec<layout::Size>,
    args: Vec<(String, Ty<'tcx>)>,
    discriminant_type_metadata: Option<&'ll DIType>,
    span: Span,
}

impl VariantMemberDescriptionFactory<'ll, 'tcx> {
    fn create_member_descriptions(&self, cx: &CodegenCx<'ll, 'tcx>)
                                      -> Vec<MemberDescription<'ll>> {
        self.args.iter().enumerate().map(|(i, &(ref name, ty))| {
            let (size, align) = cx.size_and_align_of(ty);
            MemberDescription {
                name: name.to_string(),
                type_metadata: if use_enum_fallback(cx) {
                    match self.discriminant_type_metadata {
                        // Discriminant is always the first field of our variant
                        // when using the enum fallback.
                        Some(metadata) if i == 0 => metadata,
                        _ => type_metadata(cx, ty, self.span)
                    }
                } else {
                    type_metadata(cx, ty, self.span)
                },
                offset: self.offsets[i],
                size,
                align,
                flags: DIFlags::FlagZero,
                discriminant: None,
            }
        }).collect()
    }
}

#[derive(Copy, Clone)]
enum EnumDiscriminantInfo<'ll> {
    RegularDiscriminant{ discr_field: Field, discr_type_metadata: &'ll DIType },
    OptimizedDiscriminant,
    NoDiscriminant
}

#[derive(Copy, Clone)]
enum VariantInfo<'tcx> {
    Adt(&'tcx ty::VariantDef),
    Generator(ty::GeneratorSubsts<'tcx>, &'tcx GeneratorLayout<'tcx>, VariantIdx),
}

impl<'tcx> VariantInfo<'tcx> {
    fn map_struct_name<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        match self {
            VariantInfo::Adt(variant) => f(&variant.ident.as_str()),
            VariantInfo::Generator(substs, _, variant_index) =>
                f(&substs.variant_name(*variant_index)),
        }
    }

    fn variant_name(&self) -> String {
        match self {
            VariantInfo::Adt(variant) => variant.ident.to_string(),
            VariantInfo::Generator(_, _, variant_index) => {
                // Since GDB currently prints out the raw discriminant along
                // with every variant, make each variant name be just the value
                // of the discriminant. The struct name for the variant includes
                // the actual variant description.
                format!("{}", variant_index.as_usize())
            }
        }
    }

    fn field_name(&self, i: usize) -> String {
        let field_name = match self {
            VariantInfo::Adt(variant) if variant.ctor_kind != CtorKind::Fn =>
                Some(variant.fields[i].ident.to_string()),
            VariantInfo::Generator(_, generator_layout, variant_index) => {
                let field = generator_layout.variant_fields[*variant_index][i.into()];
                let decl = &generator_layout.__local_debuginfo_codegen_only_do_not_use[field];
                decl.name.map(|name| name.to_string())
            }
            _ => None,
        };
        field_name.unwrap_or_else(|| format!("__{}", i))
    }
}

// Returns a tuple of (1) type_metadata_stub of the variant, (2) a
// MemberDescriptionFactory for producing the descriptions of the
// fields of the variant. This is a rudimentary version of a full
// RecursiveTypeDescription.
fn describe_enum_variant(
    cx: &CodegenCx<'ll, 'tcx>,
    layout: layout::TyLayout<'tcx>,
    variant: VariantInfo<'tcx>,
    discriminant_info: EnumDiscriminantInfo<'ll>,
    containing_scope: &'ll DIScope,
    span: Span,
) -> (&'ll DICompositeType, MemberDescriptionFactory<'ll, 'tcx>) {
    let metadata_stub = variant.map_struct_name(|variant_name| {
        let unique_type_id = debug_context(cx).type_map
                                              .borrow_mut()
                                              .get_unique_type_id_of_enum_variant(
                                                  cx,
                                                  layout.ty,
                                                  &variant_name);
        create_struct_stub(cx,
                           layout.ty,
                           &variant_name,
                           unique_type_id,
                           Some(containing_scope))
    });

    // Build an array of (field name, field type) pairs to be captured in the factory closure.
    let (offsets, args) = if use_enum_fallback(cx) {
        // If this is not a univariant enum, there is also the discriminant field.
        let (discr_offset, discr_arg) = match discriminant_info {
            RegularDiscriminant { discr_field, .. } => {
                // We have the layout of an enum variant, we need the layout of the outer enum
                let enum_layout = cx.layout_of(layout.ty);
                let offset = enum_layout.fields.offset(discr_field.as_usize());
                let args = (
                    "RUST$ENUM$DISR".to_owned(),
                    enum_layout.field(cx, discr_field.as_usize()).ty);
                (Some(offset), Some(args))
            }
            _ => (None, None),
        };
        (
            discr_offset.into_iter().chain((0..layout.fields.count()).map(|i| {
                layout.fields.offset(i)
            })).collect(),
            discr_arg.into_iter().chain((0..layout.fields.count()).map(|i| {
                (variant.field_name(i), layout.field(cx, i).ty)
            })).collect()
        )
    } else {
        (
            (0..layout.fields.count()).map(|i| {
                layout.fields.offset(i)
            }).collect(),
            (0..layout.fields.count()).map(|i| {
                (variant.field_name(i), layout.field(cx, i).ty)
            }).collect()
        )
    };

    let member_description_factory =
        VariantMDF(VariantMemberDescriptionFactory {
            offsets,
            args,
            discriminant_type_metadata: match discriminant_info {
                RegularDiscriminant { discr_type_metadata, .. } => {
                    Some(discr_type_metadata)
                }
                _ => None
            },
            span,
        });

    (metadata_stub, member_description_factory)
}

fn prepare_enum_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type: Ty<'tcx>,
    enum_def_id: DefId,
    unique_type_id: UniqueTypeId,
    span: Span,
    outer_field_tys: Vec<Ty<'tcx>>,
) -> RecursiveTypeDescription<'ll, 'tcx> {
    let enum_name = compute_debuginfo_type_name(cx.tcx, enum_type, false);

    let containing_scope = get_namespace_for_item(cx, enum_def_id);
    // FIXME: This should emit actual file metadata for the enum, but we
    // currently can't get the necessary information when it comes to types
    // imported from other crates. Formerly we violated the ODR when performing
    // LTO because we emitted debuginfo for the same type with varying file
    // metadata, so as a workaround we pretend that the type comes from
    // <unknown>
    let file_metadata = unknown_file_metadata(cx);

    let discriminant_type_metadata = |discr: layout::Primitive| {
        let enumerators_metadata: Vec<_> = match enum_type.sty {
            ty::Adt(def, _) => def
                .discriminants(cx.tcx)
                .zip(&def.variants)
                .map(|((_, discr), v)| {
                    let name = SmallCStr::new(&v.ident.as_str());
                    unsafe {
                        Some(llvm::LLVMRustDIBuilderCreateEnumerator(
                            DIB(cx),
                            name.as_ptr(),
                            // FIXME: what if enumeration has i128 discriminant?
                            discr.val as u64))
                    }
                })
                .collect(),
            ty::Generator(_, substs, _) => substs
                .variant_range(enum_def_id, cx.tcx)
                .map(|variant_index| {
                    let name = SmallCStr::new(&substs.variant_name(variant_index));
                    unsafe {
                        Some(llvm::LLVMRustDIBuilderCreateEnumerator(
                            DIB(cx),
                            name.as_ptr(),
                            // FIXME: what if enumeration has i128 discriminant?
                            variant_index.as_usize() as u64))
                    }
                })
                .collect(),
            _ => bug!(),
        };

        let disr_type_key = (enum_def_id, discr);
        let cached_discriminant_type_metadata = debug_context(cx).created_enum_disr_types
                                                                 .borrow()
                                                                 .get(&disr_type_key).cloned();
        match cached_discriminant_type_metadata {
            Some(discriminant_type_metadata) => discriminant_type_metadata,
            None => {
                let (discriminant_size, discriminant_align) =
                    (discr.size(cx), discr.align(cx));
                let discriminant_base_type_metadata =
                    type_metadata(cx, discr.to_ty(cx.tcx), syntax_pos::DUMMY_SP);

                let discriminant_name = match enum_type.sty {
                    ty::Adt(..) => SmallCStr::new(&cx.tcx.item_name(enum_def_id).as_str()),
                    ty::Generator(..) => SmallCStr::new(&enum_name),
                    _ => bug!(),
                };

                let discriminant_type_metadata = unsafe {
                    llvm::LLVMRustDIBuilderCreateEnumerationType(
                        DIB(cx),
                        containing_scope,
                        discriminant_name.as_ptr(),
                        file_metadata,
                        UNKNOWN_LINE_NUMBER,
                        discriminant_size.bits(),
                        discriminant_align.abi.bits() as u32,
                        create_DIArray(DIB(cx), &enumerators_metadata),
                        discriminant_base_type_metadata, true)
                };

                debug_context(cx).created_enum_disr_types
                                 .borrow_mut()
                                 .insert(disr_type_key, discriminant_type_metadata);

                discriminant_type_metadata
            }
        }
    };

    let layout = cx.layout_of(enum_type);

    match (&layout.abi, &layout.variants) {
        (&layout::Abi::Scalar(_), &layout::Variants::Multiple {
            discr_kind: layout::DiscriminantKind::Tag,
            ref discr,
            ..
        }) => return FinalMetadata(discriminant_type_metadata(discr.value)),
        _ => {}
    }

    let enum_name = SmallCStr::new(&enum_name);
    let unique_type_id_str = SmallCStr::new(
        debug_context(cx).type_map.borrow().get_unique_type_id_as_string(unique_type_id)
    );

    if use_enum_fallback(cx) {
        let discriminant_type_metadata = match layout.variants {
            layout::Variants::Single { .. } |
            layout::Variants::Multiple {
                discr_kind: layout::DiscriminantKind::Niche { .. },
                ..
            } => None,
            layout::Variants::Multiple {
                discr_kind: layout::DiscriminantKind::Tag,
                ref discr,
                ..
            } => {
                Some(discriminant_type_metadata(discr.value))
            }
        };

        let enum_metadata = unsafe {
            llvm::LLVMRustDIBuilderCreateUnionType(
                DIB(cx),
                containing_scope,
                enum_name.as_ptr(),
                file_metadata,
                UNKNOWN_LINE_NUMBER,
                layout.size.bits(),
                layout.align.abi.bits() as u32,
                DIFlags::FlagZero,
                None,
                0, // RuntimeLang
                unique_type_id_str.as_ptr())
        };

        return create_and_register_recursive_type_forward_declaration(
            cx,
            enum_type,
            unique_type_id,
            enum_metadata,
            enum_metadata,
            EnumMDF(EnumMemberDescriptionFactory {
                enum_type,
                layout,
                discriminant_type_metadata,
                containing_scope,
                span,
            }),
        );
    }

    let discriminator_name = match &enum_type.sty {
        ty::Generator(..) => Some(SmallCStr::new(&"__state")),
        _ => None,
    };
    let discriminator_name = discriminator_name.map(|n| n.as_ptr()).unwrap_or(ptr::null_mut());
    let discriminator_metadata = match layout.variants {
        // A single-variant enum has no discriminant.
        layout::Variants::Single { .. } => None,

        layout::Variants::Multiple {
            discr_kind: layout::DiscriminantKind::Niche { .. },
            ref discr,
            discr_index,
            ..
        } => {
            // Find the integer type of the correct size.
            let size = discr.value.size(cx);
            let align = discr.value.align(cx);

            let discr_type = match discr.value {
                layout::Int(t, _) => t,
                layout::Float(layout::FloatTy::F32) => Integer::I32,
                layout::Float(layout::FloatTy::F64) => Integer::I64,
                layout::Pointer => cx.data_layout().ptr_sized_integer(),
            }.to_ty(cx.tcx, false);

            let discr_metadata = basic_type_metadata(cx, discr_type);
            unsafe {
                Some(llvm::LLVMRustDIBuilderCreateMemberType(
                    DIB(cx),
                    containing_scope,
                    discriminator_name,
                    file_metadata,
                    UNKNOWN_LINE_NUMBER,
                    size.bits(),
                    align.abi.bits() as u32,
                    layout.fields.offset(discr_index).bits(),
                    DIFlags::FlagArtificial,
                    discr_metadata))
            }
        },

        layout::Variants::Multiple {
            discr_kind: layout::DiscriminantKind::Tag,
            ref discr,
            discr_index,
            ..
        } => {
            let discr_type = discr.value.to_ty(cx.tcx);
            let (size, align) = cx.size_and_align_of(discr_type);

            let discr_metadata = basic_type_metadata(cx, discr_type);
            unsafe {
                Some(llvm::LLVMRustDIBuilderCreateMemberType(
                    DIB(cx),
                    containing_scope,
                    discriminator_name,
                    file_metadata,
                    UNKNOWN_LINE_NUMBER,
                    size.bits(),
                    align.bits() as u32,
                    layout.fields.offset(discr_index).bits(),
                    DIFlags::FlagArtificial,
                    discr_metadata))
            }
        },
    };

    let mut outer_fields = match layout.variants {
        layout::Variants::Single { .. } => vec![],
        layout::Variants::Multiple { .. } => {
            let tuple_mdf = TupleMemberDescriptionFactory {
                ty: enum_type,
                component_types: outer_field_tys,
                span
            };
            tuple_mdf
                .create_member_descriptions(cx)
                .into_iter()
                .map(|desc| Some(desc.into_metadata(cx, containing_scope)))
                .collect()
        }
    };

    let variant_part_unique_type_id_str = SmallCStr::new(
        debug_context(cx).type_map
            .borrow_mut()
            .get_unique_type_id_str_of_enum_variant_part(unique_type_id)
    );
    let empty_array = create_DIArray(DIB(cx), &[]);
    let variant_part = unsafe {
        llvm::LLVMRustDIBuilderCreateVariantPart(
            DIB(cx),
            containing_scope,
            ptr::null_mut(),
            file_metadata,
            UNKNOWN_LINE_NUMBER,
            layout.size.bits(),
            layout.align.abi.bits() as u32,
            DIFlags::FlagZero,
            discriminator_metadata,
            empty_array,
            variant_part_unique_type_id_str.as_ptr())
    };
    outer_fields.push(Some(variant_part));

    // The variant part must be wrapped in a struct according to DWARF.
    let type_array = create_DIArray(DIB(cx), &outer_fields);
    let struct_wrapper = unsafe {
        llvm::LLVMRustDIBuilderCreateStructType(
            DIB(cx),
            Some(containing_scope),
            enum_name.as_ptr(),
            file_metadata,
            UNKNOWN_LINE_NUMBER,
            layout.size.bits(),
            layout.align.abi.bits() as u32,
            DIFlags::FlagZero,
            None,
            type_array,
            0,
            None,
            unique_type_id_str.as_ptr())
    };

    return create_and_register_recursive_type_forward_declaration(
        cx,
        enum_type,
        unique_type_id,
        struct_wrapper,
        variant_part,
        EnumMDF(EnumMemberDescriptionFactory {
            enum_type,
            layout,
            discriminant_type_metadata: None,
            containing_scope,
            span,
        }),
    );
}

/// Creates debug information for a composite type, that is, anything that
/// results in a LLVM struct.
///
/// Examples of Rust types to use this are: structs, tuples, boxes, vecs, and enums.
fn composite_type_metadata(
    cx: &CodegenCx<'ll, 'tcx>,
    composite_type: Ty<'tcx>,
    composite_type_name: &str,
    composite_type_unique_id: UniqueTypeId,
    member_descriptions: Vec<MemberDescription<'ll>>,
    containing_scope: Option<&'ll DIScope>,

    // Ignore source location information as long as it
    // can't be reconstructed for non-local crates.
    _file_metadata: &'ll DIFile,
    _definition_span: Span,
) -> &'ll DICompositeType {
    // Create the (empty) struct metadata node ...
    let composite_type_metadata = create_struct_stub(cx,
                                                     composite_type,
                                                     composite_type_name,
                                                     composite_type_unique_id,
                                                     containing_scope);
    // ... and immediately create and add the member descriptions.
    set_members_of_composite_type(cx,
                                  composite_type,
                                  composite_type_metadata,
                                  member_descriptions);

    composite_type_metadata
}

fn set_members_of_composite_type(cx: &CodegenCx<'ll, 'tcx>,
                                 composite_type: Ty<'tcx>,
                                 composite_type_metadata: &'ll DICompositeType,
                                 member_descriptions: Vec<MemberDescription<'ll>>) {
    // In some rare cases LLVM metadata uniquing would lead to an existing type
    // description being used instead of a new one created in
    // create_struct_stub. This would cause a hard to trace assertion in
    // DICompositeType::SetTypeArray(). The following check makes sure that we
    // get a better error message if this should happen again due to some
    // regression.
    {
        let mut composite_types_completed =
            debug_context(cx).composite_types_completed.borrow_mut();
        if composite_types_completed.contains(&composite_type_metadata) {
            bug!("debuginfo::set_members_of_composite_type() - \
                  Already completed forward declaration re-encountered.");
        } else {
            composite_types_completed.insert(composite_type_metadata);
        }
    }

    let member_metadata: Vec<_> = member_descriptions
        .into_iter()
        .map(|desc| Some(desc.into_metadata(cx, composite_type_metadata)))
        .collect();

    let type_params = compute_type_parameters(cx, composite_type);
    unsafe {
        let type_array = create_DIArray(DIB(cx), &member_metadata[..]);
        llvm::LLVMRustDICompositeTypeReplaceArrays(
            DIB(cx), composite_type_metadata, Some(type_array), type_params);
    }
}

// Compute the type parameters for a type, if any, for the given
// metadata.
fn compute_type_parameters(cx: &CodegenCx<'ll, 'tcx>, ty: Ty<'tcx>) -> Option<&'ll DIArray> {
    if let ty::Adt(def, substs) = ty.sty {
        if !substs.types().next().is_none() {
            let generics = cx.tcx.generics_of(def.did);
            let names = get_parameter_names(cx, generics);
            let template_params: Vec<_> = substs.iter().zip(names).filter_map(|(kind, name)| {
                if let UnpackedKind::Type(ty) = kind.unpack() {
                    let actual_type = cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), ty);
                    let actual_type_metadata =
                        type_metadata(cx, actual_type, syntax_pos::DUMMY_SP);
                    let name = SmallCStr::new(&name.as_str());
                    Some(unsafe {

                        Some(llvm::LLVMRustDIBuilderCreateTemplateTypeParameter(
                            DIB(cx),
                            None,
                            name.as_ptr(),
                            actual_type_metadata,
                            unknown_file_metadata(cx),
                            0,
                            0,
                        ))
                    })
                } else {
                    None
                }
            }).collect();

            return Some(create_DIArray(DIB(cx), &template_params[..]));
        }
    }
    return Some(create_DIArray(DIB(cx), &[]));

    fn get_parameter_names(cx: &CodegenCx<'_, '_>,
                           generics: &ty::Generics)
                           -> Vec<InternedString> {
        let mut names = generics.parent.map_or(vec![], |def_id| {
            get_parameter_names(cx, cx.tcx.generics_of(def_id))
        });
        names.extend(generics.params.iter().map(|param| param.name));
        names
    }
}

// A convenience wrapper around LLVMRustDIBuilderCreateStructType(). Does not do
// any caching, does not add any fields to the struct. This can be done later
// with set_members_of_composite_type().
fn create_struct_stub(
    cx: &CodegenCx<'ll, 'tcx>,
    struct_type: Ty<'tcx>,
    struct_type_name: &str,
    unique_type_id: UniqueTypeId,
    containing_scope: Option<&'ll DIScope>,
) -> &'ll DICompositeType {
    let (struct_size, struct_align) = cx.size_and_align_of(struct_type);

    let name = SmallCStr::new(struct_type_name);
    let unique_type_id = SmallCStr::new(
        debug_context(cx).type_map.borrow().get_unique_type_id_as_string(unique_type_id)
    );
    let metadata_stub = unsafe {
        // LLVMRustDIBuilderCreateStructType() wants an empty array. A null
        // pointer will lead to hard to trace and debug LLVM assertions
        // later on in llvm/lib/IR/Value.cpp.
        let empty_array = create_DIArray(DIB(cx), &[]);

        llvm::LLVMRustDIBuilderCreateStructType(
            DIB(cx),
            containing_scope,
            name.as_ptr(),
            unknown_file_metadata(cx),
            UNKNOWN_LINE_NUMBER,
            struct_size.bits(),
            struct_align.bits() as u32,
            DIFlags::FlagZero,
            None,
            empty_array,
            0,
            None,
            unique_type_id.as_ptr())
    };

    metadata_stub
}

fn create_union_stub(
    cx: &CodegenCx<'ll, 'tcx>,
    union_type: Ty<'tcx>,
    union_type_name: &str,
    unique_type_id: UniqueTypeId,
    containing_scope: &'ll DIScope,
) -> &'ll DICompositeType {
    let (union_size, union_align) = cx.size_and_align_of(union_type);

    let name = SmallCStr::new(union_type_name);
    let unique_type_id = SmallCStr::new(
        debug_context(cx).type_map.borrow().get_unique_type_id_as_string(unique_type_id)
    );
    let metadata_stub = unsafe {
        // LLVMRustDIBuilderCreateUnionType() wants an empty array. A null
        // pointer will lead to hard to trace and debug LLVM assertions
        // later on in llvm/lib/IR/Value.cpp.
        let empty_array = create_DIArray(DIB(cx), &[]);

        llvm::LLVMRustDIBuilderCreateUnionType(
            DIB(cx),
            containing_scope,
            name.as_ptr(),
            unknown_file_metadata(cx),
            UNKNOWN_LINE_NUMBER,
            union_size.bits(),
            union_align.bits() as u32,
            DIFlags::FlagZero,
            Some(empty_array),
            0, // RuntimeLang
            unique_type_id.as_ptr())
    };

    metadata_stub
}

/// Creates debug information for the given global variable.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_global_var_metadata(
    cx: &CodegenCx<'ll, '_>,
    def_id: DefId,
    global: &'ll Value,
) {
    if cx.dbg_cx.is_none() {
        return;
    }

    let tcx = cx.tcx;
    let attrs = tcx.codegen_fn_attrs(def_id);

    if attrs.flags.contains(CodegenFnAttrFlags::NO_DEBUG) {
        return;
    }

    let no_mangle = attrs.flags.contains(CodegenFnAttrFlags::NO_MANGLE);
    // We may want to remove the namespace scope if we're in an extern block, see:
    // https://github.com/rust-lang/rust/pull/46457#issuecomment-351750952
    let var_scope = get_namespace_for_item(cx, def_id);
    let span = tcx.def_span(def_id);

    let (file_metadata, line_number) = if !span.is_dummy() {
        let loc = span_start(cx, span);
        (file_metadata(cx, &loc.file.name, LOCAL_CRATE), loc.line as c_uint)
    } else {
        (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER)
    };

    let is_local_to_unit = is_node_local_to_unit(cx, def_id);
    let variable_type = Instance::mono(cx.tcx, def_id).ty(cx.tcx);
    let type_metadata = type_metadata(cx, variable_type, span);
    let var_name = SmallCStr::new(&tcx.item_name(def_id).as_str());
    let linkage_name = if no_mangle {
        None
    } else {
        let linkage_name = mangled_name_of_instance(cx, Instance::mono(tcx, def_id));
        Some(SmallCStr::new(&linkage_name.as_str()))
    };

    let global_align = cx.align_of(variable_type);

    unsafe {
        llvm::LLVMRustDIBuilderCreateStaticVariable(DIB(cx),
                                                    Some(var_scope),
                                                    var_name.as_ptr(),
                                                    // If null, linkage_name field is omitted,
                                                    // which is what we want for no_mangle statics
                                                    linkage_name.as_ref()
                                                     .map_or(ptr::null(), |name| name.as_ptr()),
                                                    file_metadata,
                                                    line_number,
                                                    type_metadata,
                                                    is_local_to_unit,
                                                    global,
                                                    None,
                                                    global_align.bytes() as u32,
        );
    }
}

/// Creates debug information for the given vtable, which is for the
/// given type.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_vtable_metadata(cx: &CodegenCx<'ll, 'tcx>, ty: Ty<'tcx>, vtable: &'ll Value) {
    if cx.dbg_cx.is_none() {
        return;
    }

    let type_metadata = type_metadata(cx, ty, syntax_pos::DUMMY_SP);

    unsafe {
        // LLVMRustDIBuilderCreateStructType() wants an empty array. A null
        // pointer will lead to hard to trace and debug LLVM assertions
        // later on in llvm/lib/IR/Value.cpp.
        let empty_array = create_DIArray(DIB(cx), &[]);

        let name = const_cstr!("vtable");

        // Create a new one each time.  We don't want metadata caching
        // here, because each vtable will refer to a unique containing
        // type.
        let vtable_type = llvm::LLVMRustDIBuilderCreateStructType(
            DIB(cx),
            NO_SCOPE_METADATA,
            name.as_ptr(),
            unknown_file_metadata(cx),
            UNKNOWN_LINE_NUMBER,
            Size::ZERO.bits(),
            cx.tcx.data_layout.pointer_align.abi.bits() as u32,
            DIFlags::FlagArtificial,
            None,
            empty_array,
            0,
            Some(type_metadata),
            name.as_ptr()
        );

        llvm::LLVMRustDIBuilderCreateStaticVariable(DIB(cx),
                                                    NO_SCOPE_METADATA,
                                                    name.as_ptr(),
                                                    ptr::null(),
                                                    unknown_file_metadata(cx),
                                                    UNKNOWN_LINE_NUMBER,
                                                    vtable_type,
                                                    true,
                                                    vtable,
                                                    None,
                                                    0);
    }
}

// Creates an "extension" of an existing DIScope into another file.
pub fn extend_scope_to_file(
    cx: &CodegenCx<'ll, '_>,
    scope_metadata: &'ll DIScope,
    file: &syntax_pos::SourceFile,
    defining_crate: CrateNum,
) -> &'ll DILexicalBlock {
    let file_metadata = file_metadata(cx, &file.name, defining_crate);
    unsafe {
        llvm::LLVMRustDIBuilderCreateLexicalBlockFile(
            DIB(cx),
            scope_metadata,
            file_metadata)
    }
}
