/// Supports normalizing a type prior to encoding
use rustc_middle::ty::fold::{TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{self, IntTy, Ty, TyCtxt, UintTy};
use rustc_span::sym;

use crate::typeid;

/// Normalizes a type to a form suitable for encoding.
///
/// Always:
///
/// * `c_void` -> `()`
/// * Unwrap `#[repr(transparent)]` (when possible)
/// * Erases projection clauses (This behavior is undesirable and to be removed)
/// * Erases some arguments from trait clauses (To be adjusted)
///
/// With NORMALIZE_INTEGERS:
///
/// * `bool` -> `u8`
/// * `char` -> `u32`
/// * `isize` / `usize` -> `iK` / `uK` where K is platform pointer width
///
/// With GENERALIZE_POINTERS:
///
/// * `*mut T` / `*const T` -> `*mut ()` / `*const ()`
/// * `&mut T` / `&T` -> `&mut ()` / `&()`
/// * `fn(..) -> T` -> `*const ()`
pub fn transform<'tcx>(tcx: TyCtxt<'tcx>, options: typeid::Options, ty: Ty<'tcx>) -> Ty<'tcx> {
    TransformTy::new(tcx, options).fold_ty(ty)
}

struct TransformTy<'tcx> {
    tcx: TyCtxt<'tcx>,
    options: typeid::Options,
    parents: Vec<Ty<'tcx>>,
}

impl<'tcx> TransformTy<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, options: typeid::Options) -> Self {
        TransformTy { tcx, options, parents: Vec::new() }
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for TransformTy<'tcx> {
    // Transforms a ty:Ty for being encoded and used in the substitution dictionary. It transforms
    // all c_void types into unit types unconditionally, generalizes pointers if
    // Options::GENERALIZE_POINTERS option is set, and normalizes integers if
    // Options::NORMALIZE_INTEGERS option is set.
    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.kind() {
            ty::Array(..)
            | ty::Closure(..)
            | ty::Coroutine(..)
            | ty::CoroutineClosure(..)
            | ty::CoroutineWitness(..)
            | ty::Float(..)
            | ty::FnDef(..)
            | ty::Foreign(..)
            | ty::Never
            | ty::Slice(..)
            | ty::Str
            | ty::Tuple(..) => t.super_fold_with(self),

            ty::Bool => {
                if self.options.contains(typeid::Options::NORMALIZE_INTEGERS) {
                    // Note: on all platforms that Rust's currently supports, its size and alignment
                    // are 1, and its ABI class is INTEGER - see Rust Layout and ABIs.
                    //
                    // (See https://rust-lang.github.io/unsafe-code-guidelines/layout/scalars.html#bool.)
                    //
                    // Clang represents bool as an 8-bit unsigned integer.
                    self.tcx.types.u8
                } else {
                    t
                }
            }

            ty::Char => {
                if self.options.contains(typeid::Options::NORMALIZE_INTEGERS) {
                    // Since #118032, char is guaranteed to have the same size, alignment, and
                    // function call ABI as u32 on all platforms.
                    self.tcx.types.u32
                } else {
                    t
                }
            }

            ty::Int(..) | ty::Uint(..) => {
                if self.options.contains(typeid::Options::NORMALIZE_INTEGERS) {
                    // Note: C99 7.18.2.4 requires uintptr_t and intptr_t to be at least 16-bit
                    // wide. All platforms we currently support have a C platform, and as a
                    // consequence, isize/usize are at least 16-bit wide for all of them.
                    //
                    // (See https://rust-lang.github.io/unsafe-code-guidelines/layout/scalars.html#isize-and-usize.)
                    match t.kind() {
                        ty::Int(IntTy::Isize) => match self.tcx.sess.target.pointer_width {
                            16 => self.tcx.types.i16,
                            32 => self.tcx.types.i32,
                            64 => self.tcx.types.i64,
                            128 => self.tcx.types.i128,
                            _ => bug!(
                                "fold_ty: unexpected pointer width `{}`",
                                self.tcx.sess.target.pointer_width
                            ),
                        },
                        ty::Uint(UintTy::Usize) => match self.tcx.sess.target.pointer_width {
                            16 => self.tcx.types.u16,
                            32 => self.tcx.types.u32,
                            64 => self.tcx.types.u64,
                            128 => self.tcx.types.u128,
                            _ => bug!(
                                "fold_ty: unexpected pointer width `{}`",
                                self.tcx.sess.target.pointer_width
                            ),
                        },
                        _ => t,
                    }
                } else {
                    t
                }
            }

            ty::Adt(..) if t.is_c_void(self.tcx) => self.tcx.types.unit,

            ty::Adt(adt_def, args) => {
                if adt_def.repr().transparent() && adt_def.is_struct() && !self.parents.contains(&t)
                {
                    // Don't transform repr(transparent) types with an user-defined CFI encoding to
                    // preserve the user-defined CFI encoding.
                    if self.tcx.get_attr(adt_def.did(), sym::cfi_encoding).is_some() {
                        return t;
                    }
                    let variant = adt_def.non_enum_variant();
                    let param_env = self.tcx.param_env(variant.def_id);
                    let field = variant.fields.iter().find(|field| {
                        let ty = self.tcx.type_of(field.did).instantiate_identity();
                        let is_zst = self
                            .tcx
                            .layout_of(param_env.and(ty))
                            .is_ok_and(|layout| layout.is_zst());
                        !is_zst
                    });
                    if let Some(field) = field {
                        let ty0 = self.tcx.type_of(field.did).instantiate(self.tcx, args);
                        // Generalize any repr(transparent) user-defined type that is either a
                        // pointer or reference, and either references itself or any other type that
                        // contains or references itself, to avoid a reference cycle.

                        // If the self reference is not through a pointer, for example, due
                        // to using `PhantomData`, need to skip normalizing it if we hit it again.
                        self.parents.push(t);
                        let ty = if ty0.is_any_ptr() && ty0.contains(t) {
                            let options = self.options;
                            self.options |= typeid::Options::GENERALIZE_POINTERS;
                            let ty = ty0.fold_with(self);
                            self.options = options;
                            ty
                        } else {
                            ty0.fold_with(self)
                        };
                        self.parents.pop();
                        ty
                    } else {
                        // Transform repr(transparent) types without non-ZST field into ()
                        self.tcx.types.unit
                    }
                } else {
                    t.super_fold_with(self)
                }
            }

            ty::Ref(..) => {
                if self.options.contains(typeid::Options::GENERALIZE_POINTERS) {
                    if t.is_mutable_ptr() {
                        Ty::new_mut_ref(self.tcx, self.tcx.lifetimes.re_static, self.tcx.types.unit)
                    } else {
                        Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_static, self.tcx.types.unit)
                    }
                } else {
                    t.super_fold_with(self)
                }
            }

            ty::RawPtr(..) => {
                if self.options.contains(typeid::Options::GENERALIZE_POINTERS) {
                    if t.is_mutable_ptr() {
                        Ty::new_mut_ptr(self.tcx, self.tcx.types.unit)
                    } else {
                        Ty::new_imm_ptr(self.tcx, self.tcx.types.unit)
                    }
                } else {
                    t.super_fold_with(self)
                }
            }

            ty::FnPtr(..) => {
                if self.options.contains(typeid::Options::GENERALIZE_POINTERS) {
                    Ty::new_imm_ptr(self.tcx, self.tcx.types.unit)
                } else {
                    t.super_fold_with(self)
                }
            }

            ty::Dynamic(predicates, _region, kind) => {
                let predicates = self.tcx.mk_poly_existential_predicates_from_iter(
                    predicates.iter().filter_map(|predicate| match predicate.skip_binder() {
                        ty::ExistentialPredicate::Trait(trait_ref) => {
                            let trait_ref = ty::TraitRef::identity(self.tcx, trait_ref.def_id);
                            Some(ty::Binder::dummy(ty::ExistentialPredicate::Trait(
                                ty::ExistentialTraitRef::erase_self_ty(self.tcx, trait_ref),
                            )))
                        }
                        ty::ExistentialPredicate::Projection(..) => None,
                        ty::ExistentialPredicate::AutoTrait(..) => Some(predicate),
                    }),
                );

                Ty::new_dynamic(self.tcx, predicates, self.tcx.lifetimes.re_erased, *kind)
            }

            ty::Alias(..) => {
                self.fold_ty(self.tcx.normalize_erasing_regions(ty::ParamEnv::reveal_all(), t))
            }

            ty::Bound(..) | ty::Error(..) | ty::Infer(..) | ty::Param(..) | ty::Placeholder(..) => {
                bug!("fold_ty: unexpected `{:?}`", t.kind());
            }
        }
    }

    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}
