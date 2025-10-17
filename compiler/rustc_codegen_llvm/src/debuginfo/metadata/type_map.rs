use std::cell::RefCell;

use libc::c_uint;
use rustc_abi::{Align, Size, VariantIdx};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_macros::HashStable;
use rustc_middle::bug;
use rustc_middle::ty::{self, ExistentialTraitRef, Ty, TyCtxt};

use super::{DefinitionLocation, SmallVec, UNKNOWN_LINE_NUMBER, unknown_file_metadata};
use crate::common::CodegenCx;
use crate::debuginfo::utils::{DIB, create_DIArray, debug_context};
use crate::llvm::debuginfo::{DIFlags, DIScope, DIType};
use crate::llvm::{self};

mod private {
    use rustc_macros::HashStable;

    // This type cannot be constructed outside of this module because
    // it has a private field. We make use of this in order to prevent
    // `UniqueTypeId` from being constructed directly, without asserting
    // the preconditions.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, HashStable)]
    pub(crate) struct HiddenZst;
}

/// A unique identifier for anything that we create a debuginfo node for.
/// The types it contains are expected to already be normalized (which
/// is asserted in the constructors).
///
/// Note that there are some things that only show up in debuginfo, like
/// the separate type descriptions for each enum variant. These get an ID
/// too because they have their own debuginfo node in LLVM IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, HashStable)]
pub(super) enum UniqueTypeId<'tcx> {
    /// The ID of a regular type as it shows up at the language level.
    Ty(Ty<'tcx>, private::HiddenZst),
    /// The ID for the single DW_TAG_variant_part nested inside the top-level
    /// DW_TAG_structure_type that describes enums and coroutines.
    VariantPart(Ty<'tcx>, private::HiddenZst),
    /// The ID for the artificial struct type describing a single enum variant.
    VariantStructType(Ty<'tcx>, VariantIdx, private::HiddenZst),
    /// The ID for the additional wrapper struct type describing an enum variant in CPP-like mode.
    VariantStructTypeCppLikeWrapper(Ty<'tcx>, VariantIdx, private::HiddenZst),
    /// The ID of the artificial type we create for VTables.
    VTableTy(Ty<'tcx>, Option<ExistentialTraitRef<'tcx>>, private::HiddenZst),
}

impl<'tcx> UniqueTypeId<'tcx> {
    pub(crate) fn for_ty(tcx: TyCtxt<'tcx>, t: Ty<'tcx>) -> Self {
        assert_eq!(t, tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), t));
        UniqueTypeId::Ty(t, private::HiddenZst)
    }

    pub(crate) fn for_enum_variant_part(tcx: TyCtxt<'tcx>, enum_ty: Ty<'tcx>) -> Self {
        assert_eq!(
            enum_ty,
            tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), enum_ty)
        );
        UniqueTypeId::VariantPart(enum_ty, private::HiddenZst)
    }

    pub(crate) fn for_enum_variant_struct_type(
        tcx: TyCtxt<'tcx>,
        enum_ty: Ty<'tcx>,
        variant_idx: VariantIdx,
    ) -> Self {
        assert_eq!(
            enum_ty,
            tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), enum_ty)
        );
        UniqueTypeId::VariantStructType(enum_ty, variant_idx, private::HiddenZst)
    }

    pub(crate) fn for_enum_variant_struct_type_wrapper(
        tcx: TyCtxt<'tcx>,
        enum_ty: Ty<'tcx>,
        variant_idx: VariantIdx,
    ) -> Self {
        assert_eq!(
            enum_ty,
            tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), enum_ty)
        );
        UniqueTypeId::VariantStructTypeCppLikeWrapper(enum_ty, variant_idx, private::HiddenZst)
    }

    pub(crate) fn for_vtable_ty(
        tcx: TyCtxt<'tcx>,
        self_type: Ty<'tcx>,
        implemented_trait: Option<ExistentialTraitRef<'tcx>>,
    ) -> Self {
        assert_eq!(
            self_type,
            tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), self_type)
        );
        assert_eq!(
            implemented_trait,
            tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), implemented_trait)
        );
        UniqueTypeId::VTableTy(self_type, implemented_trait, private::HiddenZst)
    }

    /// Generates a string version of this [UniqueTypeId], which can be used as the `UniqueId`
    /// argument of the various `LLVMRustDIBuilderCreate*Type()` methods.
    ///
    /// Right now this takes the form of a hex-encoded opaque hash value.
    fn generate_unique_id_string(self, tcx: TyCtxt<'tcx>) -> String {
        let mut hasher = StableHasher::new();
        tcx.with_stable_hashing_context(|mut hcx| {
            hcx.while_hashing_spans(false, |hcx| self.hash_stable(hcx, &mut hasher))
        });
        hasher.finish::<Fingerprint>().to_hex()
    }

    pub(crate) fn expect_ty(self) -> Ty<'tcx> {
        match self {
            UniqueTypeId::Ty(ty, _) => ty,
            _ => bug!("Expected `UniqueTypeId::Ty` but found `{:?}`", self),
        }
    }
}

/// The `TypeMap` is where the debug context holds the type metadata nodes
/// created so far. The debuginfo nodes are identified by `UniqueTypeId`.
#[derive(Default)]
pub(crate) struct TypeMap<'ll, 'tcx> {
    pub(super) unique_id_to_di_node: RefCell<FxHashMap<UniqueTypeId<'tcx>, &'ll DIType>>,
}

impl<'ll, 'tcx> TypeMap<'ll, 'tcx> {
    /// Adds a `UniqueTypeId` to metadata mapping to the `TypeMap`. The method will
    /// fail if the mapping already exists.
    pub(super) fn insert(&self, unique_type_id: UniqueTypeId<'tcx>, metadata: &'ll DIType) {
        if self.unique_id_to_di_node.borrow_mut().insert(unique_type_id, metadata).is_some() {
            bug!("type metadata for unique ID '{:?}' is already in the `TypeMap`!", unique_type_id);
        }
    }

    pub(super) fn di_node_for_unique_id(
        &self,
        unique_type_id: UniqueTypeId<'tcx>,
    ) -> Option<&'ll DIType> {
        self.unique_id_to_di_node.borrow().get(&unique_type_id).cloned()
    }
}

pub(crate) struct DINodeCreationResult<'ll> {
    pub di_node: &'ll DIType,
    pub already_stored_in_typemap: bool,
}

impl<'ll> DINodeCreationResult<'ll> {
    pub(crate) fn new(di_node: &'ll DIType, already_stored_in_typemap: bool) -> Self {
        DINodeCreationResult { di_node, already_stored_in_typemap }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Stub<'ll> {
    Struct,
    Union,
    VTableTy { vtable_holder: &'ll DIType },
}

pub(crate) struct StubInfo<'ll, 'tcx> {
    metadata: &'ll DIType,
    unique_type_id: UniqueTypeId<'tcx>,
}

impl<'ll, 'tcx> StubInfo<'ll, 'tcx> {
    pub(super) fn new(
        cx: &CodegenCx<'ll, 'tcx>,
        unique_type_id: UniqueTypeId<'tcx>,
        build: impl FnOnce(&CodegenCx<'ll, 'tcx>, /* unique_type_id_str: */ &str) -> &'ll DIType,
    ) -> StubInfo<'ll, 'tcx> {
        let unique_type_id_str = unique_type_id.generate_unique_id_string(cx.tcx);
        let di_node = build(cx, &unique_type_id_str);
        StubInfo { metadata: di_node, unique_type_id }
    }
}

/// Create a stub debuginfo node onto which fields and nested types can be attached.
pub(super) fn stub<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    kind: Stub<'ll>,
    unique_type_id: UniqueTypeId<'tcx>,
    name: &str,
    def_location: Option<DefinitionLocation<'ll>>,
    (size, align): (Size, Align),
    containing_scope: Option<&'ll DIScope>,
    flags: DIFlags,
) -> StubInfo<'ll, 'tcx> {
    let no_elements: &[Option<&llvm::Metadata>] = &[];
    let unique_type_id_str = unique_type_id.generate_unique_id_string(cx.tcx);

    let (file_metadata, line_number) = if let Some(def_location) = def_location {
        (def_location.0, def_location.1)
    } else {
        (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER)
    };

    let metadata = match kind {
        Stub::Struct | Stub::VTableTy { .. } => {
            let vtable_holder = match kind {
                Stub::VTableTy { vtable_holder } => Some(vtable_holder),
                _ => None,
            };
            unsafe {
                llvm::LLVMDIBuilderCreateStructType(
                    DIB(cx),
                    containing_scope,
                    name.as_ptr(),
                    name.len(),
                    file_metadata,
                    line_number,
                    size.bits(),
                    align.bits() as u32,
                    flags,
                    None,
                    no_elements.as_ptr(),
                    no_elements.len() as c_uint,
                    0u32, // (Objective-C runtime version; default is 0)
                    vtable_holder,
                    unique_type_id_str.as_ptr(),
                    unique_type_id_str.len(),
                )
            }
        }
        Stub::Union => unsafe {
            llvm::LLVMDIBuilderCreateUnionType(
                DIB(cx),
                containing_scope,
                name.as_ptr(),
                name.len(),
                file_metadata,
                line_number,
                size.bits(),
                align.bits() as u32,
                flags,
                no_elements.as_ptr(),
                no_elements.len() as c_uint,
                0u32, // (Objective-C runtime version; default is 0)
                unique_type_id_str.as_ptr(),
                unique_type_id_str.len(),
            )
        },
    };
    StubInfo { metadata, unique_type_id }
}

struct AdtStackPopGuard<'ll, 'tcx, 'a> {
    cx: &'a CodegenCx<'ll, 'tcx>,
}

impl<'ll, 'tcx, 'a> Drop for AdtStackPopGuard<'ll, 'tcx, 'a> {
    fn drop(&mut self) {
        debug_context(self.cx).adt_stack.borrow_mut().pop();
    }
}

/// This function enables creating debuginfo nodes that can recursively refer to themselves.
/// It will first insert the given stub into the type map and only then execute the `members`
/// and `generics` closures passed in. These closures have access to the stub so they can
/// directly attach fields to them. If the type of a field transitively refers back
/// to the type currently being built, the stub will already be found in the type map,
/// which effectively breaks the recursion cycle.
pub(super) fn build_type_with_children<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    stub_info: StubInfo<'ll, 'tcx>,
    members: impl FnOnce(&CodegenCx<'ll, 'tcx>, &'ll DIType) -> SmallVec<&'ll DIType>,
    generics: impl FnOnce(&CodegenCx<'ll, 'tcx>) -> SmallVec<Option<&'ll DIType>>,
) -> DINodeCreationResult<'ll> {
    assert_eq!(debug_context(cx).type_map.di_node_for_unique_id(stub_info.unique_type_id), None);

    let mut _adt_stack_pop_guard = None;
    if let UniqueTypeId::Ty(ty, ..) = stub_info.unique_type_id
        && let ty::Adt(adt_def, args) = ty.kind()
    {
        let def_id = adt_def.did();
        // If any child type references the original type definition and the child type has a type
        // parameter that strictly contains the original parameter, the original type is a recursive
        // type that can expanding indefinitely. Example,
        // ```
        // enum Recursive<T> {
        //     Recurse(*const Recursive<Wrap<T>>),
        //     Item(T),
        // }
        // ```
        let is_expanding_recursive = {
            let stack = debug_context(cx).adt_stack.borrow();
            stack
                .iter()
                .enumerate()
                .rev()
                .skip(1)
                .filter(|(_, (ancestor_def_id, _))| def_id == *ancestor_def_id)
                .any(|(ancestor_index, (_, ancestor_args))| {
                    args.iter()
                        .zip(ancestor_args.iter())
                        .filter_map(|(arg, ancestor_arg)| arg.as_type().zip(ancestor_arg.as_type()))
                        .any(|(arg, ancestor_arg)|
                            // Strictly contains.
                            (arg != ancestor_arg && arg.contains(ancestor_arg))
                            // Check all types between current and ancestor use the
                            // ancestor_arg.
                            // Otherwise, duplicate wrappers in normal recursive type may be
                            // regarded as expanding.
                            // ```
                            // struct Recursive {
                            //     a: Box<Box<Recursive>>,
                            // }
                            // ```
                            // It can produce an ADT stack like this,
                            // - Box<Recursive>
                            // - Recursive
                            // - Box<Box<Recursive>>
                            && stack[ancestor_index + 1..stack.len()].iter().all(
                                |(_, intermediate_args)|
                                    intermediate_args
                                        .iter()
                                        .filter_map(|arg| arg.as_type())
                                        .any(|mid_arg| mid_arg.contains(ancestor_arg))
                            ))
                })
        };
        if is_expanding_recursive {
            // FIXME: indicate that this is an expanding recursive type in stub metadata?
            return DINodeCreationResult::new(stub_info.metadata, false);
        } else {
            debug_context(cx).adt_stack.borrow_mut().push((def_id, args));
            _adt_stack_pop_guard = Some(AdtStackPopGuard { cx });
        }
    }

    debug_context(cx).type_map.insert(stub_info.unique_type_id, stub_info.metadata);

    let members: SmallVec<_> =
        members(cx, stub_info.metadata).into_iter().map(|node| Some(node)).collect();
    let generics = generics(cx);

    if !(members.is_empty() && generics.is_empty()) {
        unsafe {
            let members_array = create_DIArray(DIB(cx), &members[..]);
            let generics_array = create_DIArray(DIB(cx), &generics[..]);
            llvm::LLVMRustDICompositeTypeReplaceArrays(
                DIB(cx),
                stub_info.metadata,
                Some(members_array),
                Some(generics_array),
            );
        }
    }

    DINodeCreationResult { di_node: stub_info.metadata, already_stored_in_typemap: true }
}
