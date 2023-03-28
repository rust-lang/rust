use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_index::vec::Idx;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::*;
use rustc_middle::traits::Reveal;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_target::abi::{FieldIdx, VariantIdx, FIRST_VARIANT};
use std::{fmt, iter};

/// The value of an inserted drop flag.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum DropFlagState {
    /// The tracked value is initialized and needs to be dropped when leaving its scope.
    Present,

    /// The tracked value is uninitialized or was moved out of and does not need to be dropped when
    /// leaving its scope.
    Absent,
}

impl DropFlagState {
    pub fn value(self) -> bool {
        match self {
            DropFlagState::Present => true,
            DropFlagState::Absent => false,
        }
    }
}

/// Describes how/if a value should be dropped.
#[derive(Debug)]
pub enum DropStyle {
    /// The value is already dead at the drop location, no drop will be executed.
    Dead,

    /// The value is known to always be initialized at the drop location, drop will always be
    /// executed.
    Static,

    /// Whether the value needs to be dropped depends on its drop flag.
    Conditional,

    /// An "open" drop is one where only the fields of a value are dropped.
    ///
    /// For example, this happens when moving out of a struct field: The rest of the struct will be
    /// dropped in such an "open" drop. It is also used to generate drop glue for the individual
    /// components of a value, for example for dropping array elements.
    Open,
}

/// Which drop flags to affect/check with an operation.
#[derive(Debug)]
pub enum DropFlagMode {
    /// Only affect the top-level drop flag, not that of any contained fields.
    Shallow,
    /// Affect all nested drop flags in addition to the top-level one.
    Deep,
}

/// Describes if unwinding is necessary and where to unwind to if a panic occurs.
#[derive(Copy, Clone, Debug)]
pub enum Unwind {
    /// Unwind to this block.
    To(BasicBlock),
    /// Already in an unwind path, any panic will cause an abort.
    InCleanup,
}

impl Unwind {
    fn is_cleanup(self) -> bool {
        match self {
            Unwind::To(..) => false,
            Unwind::InCleanup => true,
        }
    }

    fn into_option(self) -> Option<BasicBlock> {
        match self {
            Unwind::To(bb) => Some(bb),
            Unwind::InCleanup => None,
        }
    }

    fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(BasicBlock) -> BasicBlock,
    {
        match self {
            Unwind::To(bb) => Unwind::To(f(bb)),
            Unwind::InCleanup => Unwind::InCleanup,
        }
    }
}

pub trait DropElaborator<'a, 'tcx>: fmt::Debug {
    /// The type representing paths that can be moved out of.
    ///
    /// Users can move out of individual fields of a struct, such as `a.b.c`. This type is used to
    /// represent such move paths. Sometimes tracking individual move paths is not necessary, in
    /// which case this may be set to (for example) `()`.
    type Path: Copy + fmt::Debug;

    // Accessors

    fn patch(&mut self) -> &mut MirPatch<'tcx>;
    fn body(&self) -> &'a Body<'tcx>;
    fn tcx(&self) -> TyCtxt<'tcx>;
    fn param_env(&self) -> ty::ParamEnv<'tcx>;

    // Drop logic

    /// Returns how `path` should be dropped, given `mode`.
    fn drop_style(&self, path: Self::Path, mode: DropFlagMode) -> DropStyle;

    /// Returns the drop flag of `path` as a MIR `Operand` (or `None` if `path` has no drop flag).
    fn get_drop_flag(&mut self, path: Self::Path) -> Option<Operand<'tcx>>;

    /// Modifies the MIR patch so that the drop flag of `path` (if any) is cleared at `location`.
    ///
    /// If `mode` is deep, drop flags of all child paths should also be cleared by inserting
    /// additional statements.
    fn clear_drop_flag(&mut self, location: Location, path: Self::Path, mode: DropFlagMode);

    // Subpaths

    /// Returns the subpath of a field of `path` (or `None` if there is no dedicated subpath).
    ///
    /// If this returns `None`, `field` will not get a dedicated drop flag.
    fn field_subpath(&self, path: Self::Path, field: FieldIdx) -> Option<Self::Path>;

    /// Returns the subpath of a dereference of `path` (or `None` if there is no dedicated subpath).
    ///
    /// If this returns `None`, `*path` will not get a dedicated drop flag.
    ///
    /// This is only relevant for `Box<T>`, where the contained `T` can be moved out of the box.
    fn deref_subpath(&self, path: Self::Path) -> Option<Self::Path>;

    /// Returns the subpath of downcasting `path` to one of its variants.
    ///
    /// If this returns `None`, the downcast of `path` will not get a dedicated drop flag.
    fn downcast_subpath(&self, path: Self::Path, variant: VariantIdx) -> Option<Self::Path>;

    /// Returns the subpath of indexing a fixed-size array `path`.
    ///
    /// If this returns `None`, elements of `path` will not get a dedicated drop flag.
    ///
    /// This is only relevant for array patterns, which can move out of individual array elements.
    fn array_subpath(&self, path: Self::Path, index: u64, size: u64) -> Option<Self::Path>;
}

#[derive(Debug)]
struct DropCtxt<'l, 'b, 'tcx, D>
where
    D: DropElaborator<'b, 'tcx>,
{
    elaborator: &'l mut D,

    source_info: SourceInfo,

    place: Place<'tcx>,
    path: D::Path,
    succ: BasicBlock,
    unwind: Unwind,
}

/// "Elaborates" a drop of `place`/`path` and patches `bb`'s terminator to execute it.
///
/// The passed `elaborator` is used to determine what should happen at the drop terminator. It
/// decides whether the drop can be statically determined or whether it needs a dynamic drop flag,
/// and whether the drop is "open", ie. should be expanded to drop all subfields of the dropped
/// value.
///
/// When this returns, the MIR patch in the `elaborator` contains the necessary changes.
pub fn elaborate_drop<'b, 'tcx, D>(
    elaborator: &mut D,
    source_info: SourceInfo,
    place: Place<'tcx>,
    path: D::Path,
    succ: BasicBlock,
    unwind: Unwind,
    bb: BasicBlock,
) where
    D: DropElaborator<'b, 'tcx>,
    'tcx: 'b,
{
    DropCtxt { elaborator, source_info, place, path, succ, unwind }.elaborate_drop(bb)
}

impl<'l, 'b, 'tcx, D> DropCtxt<'l, 'b, 'tcx, D>
where
    D: DropElaborator<'b, 'tcx>,
    'tcx: 'b,
{
    fn place_ty(&self, place: Place<'tcx>) -> Ty<'tcx> {
        place.ty(self.elaborator.body(), self.tcx()).ty
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.elaborator.tcx()
    }

    /// This elaborates a single drop instruction, located at `bb`, and
    /// patches over it.
    ///
    /// The elaborated drop checks the drop flags to only drop what
    /// is initialized.
    ///
    /// In addition, the relevant drop flags also need to be cleared
    /// to avoid double-drops. However, in the middle of a complex
    /// drop, one must avoid clearing some of the flags before they
    /// are read, as that would cause a memory leak.
    ///
    /// In particular, when dropping an ADT, multiple fields may be
    /// joined together under the `rest` subpath. They are all controlled
    /// by the primary drop flag, but only the last rest-field dropped
    /// should clear it (and it must also not clear anything else).
    //
    // FIXME: I think we should just control the flags externally,
    // and then we do not need this machinery.
    pub fn elaborate_drop(&mut self, bb: BasicBlock) {
        debug!("elaborate_drop({:?}, {:?})", bb, self);
        let style = self.elaborator.drop_style(self.path, DropFlagMode::Deep);
        debug!("elaborate_drop({:?}, {:?}): live - {:?}", bb, self, style);
        match style {
            DropStyle::Dead => {
                self.elaborator
                    .patch()
                    .patch_terminator(bb, TerminatorKind::Goto { target: self.succ });
            }
            DropStyle::Static => {
                self.elaborator.patch().patch_terminator(
                    bb,
                    TerminatorKind::Drop {
                        place: self.place,
                        target: self.succ,
                        unwind: self.unwind.into_option(),
                    },
                );
            }
            DropStyle::Conditional => {
                let drop_bb = self.complete_drop(self.succ, self.unwind);
                self.elaborator
                    .patch()
                    .patch_terminator(bb, TerminatorKind::Goto { target: drop_bb });
            }
            DropStyle::Open => {
                let drop_bb = self.open_drop();
                self.elaborator
                    .patch()
                    .patch_terminator(bb, TerminatorKind::Goto { target: drop_bb });
            }
        }
    }

    /// Returns the place and move path for each field of `variant`,
    /// (the move path is `None` if the field is a rest field).
    fn move_paths_for_fields(
        &self,
        base_place: Place<'tcx>,
        variant_path: D::Path,
        variant: &'tcx ty::VariantDef,
        substs: SubstsRef<'tcx>,
    ) -> Vec<(Place<'tcx>, Option<D::Path>)> {
        variant
            .fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let field = FieldIdx::new(i);
                let subpath = self.elaborator.field_subpath(variant_path, field);
                let tcx = self.tcx();

                assert_eq!(self.elaborator.param_env().reveal(), Reveal::All);
                let field_ty =
                    tcx.normalize_erasing_regions(self.elaborator.param_env(), f.ty(tcx, substs));
                (tcx.mk_place_field(base_place, field, field_ty), subpath)
            })
            .collect()
    }

    fn drop_subpath(
        &mut self,
        place: Place<'tcx>,
        path: Option<D::Path>,
        succ: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        if let Some(path) = path {
            debug!("drop_subpath: for std field {:?}", place);

            DropCtxt {
                elaborator: self.elaborator,
                source_info: self.source_info,
                path,
                place,
                succ,
                unwind,
            }
            .elaborated_drop_block()
        } else {
            debug!("drop_subpath: for rest field {:?}", place);

            DropCtxt {
                elaborator: self.elaborator,
                source_info: self.source_info,
                place,
                succ,
                unwind,
                // Using `self.path` here to condition the drop on
                // our own drop flag.
                path: self.path,
            }
            .complete_drop(succ, unwind)
        }
    }

    /// Creates one-half of the drop ladder for a list of fields, and return
    /// the list of steps in it in reverse order, with the first step
    /// dropping 0 fields and so on.
    ///
    /// `unwind_ladder` is such a list of steps in reverse order,
    /// which is called if the matching step of the drop glue panics.
    fn drop_halfladder(
        &mut self,
        unwind_ladder: &[Unwind],
        mut succ: BasicBlock,
        fields: &[(Place<'tcx>, Option<D::Path>)],
    ) -> Vec<BasicBlock> {
        iter::once(succ)
            .chain(fields.iter().rev().zip(unwind_ladder).map(|(&(place, path), &unwind_succ)| {
                succ = self.drop_subpath(place, path, succ, unwind_succ);
                succ
            }))
            .collect()
    }

    fn drop_ladder_bottom(&mut self) -> (BasicBlock, Unwind) {
        // Clear the "master" drop flag at the end. This is needed
        // because the "master" drop protects the ADT's discriminant,
        // which is invalidated after the ADT is dropped.
        (self.drop_flag_reset_block(DropFlagMode::Shallow, self.succ, self.unwind), self.unwind)
    }

    /// Creates a full drop ladder, consisting of 2 connected half-drop-ladders
    ///
    /// For example, with 3 fields, the drop ladder is
    ///
    /// .d0:
    ///     ELAB(drop location.0 [target=.d1, unwind=.c1])
    /// .d1:
    ///     ELAB(drop location.1 [target=.d2, unwind=.c2])
    /// .d2:
    ///     ELAB(drop location.2 [target=`self.succ`, unwind=`self.unwind`])
    /// .c1:
    ///     ELAB(drop location.1 [target=.c2])
    /// .c2:
    ///     ELAB(drop location.2 [target=`self.unwind`])
    ///
    /// NOTE: this does not clear the master drop flag, so you need
    /// to point succ/unwind on a `drop_ladder_bottom`.
    fn drop_ladder(
        &mut self,
        fields: Vec<(Place<'tcx>, Option<D::Path>)>,
        succ: BasicBlock,
        unwind: Unwind,
    ) -> (BasicBlock, Unwind) {
        debug!("drop_ladder({:?}, {:?})", self, fields);

        let mut fields = fields;
        fields.retain(|&(place, _)| {
            self.place_ty(place).needs_drop(self.tcx(), self.elaborator.param_env())
        });

        debug!("drop_ladder - fields needing drop: {:?}", fields);

        let unwind_ladder = vec![Unwind::InCleanup; fields.len() + 1];
        let unwind_ladder: Vec<_> = if let Unwind::To(target) = unwind {
            let halfladder = self.drop_halfladder(&unwind_ladder, target, &fields);
            halfladder.into_iter().map(Unwind::To).collect()
        } else {
            unwind_ladder
        };

        let normal_ladder = self.drop_halfladder(&unwind_ladder, succ, &fields);

        (*normal_ladder.last().unwrap(), *unwind_ladder.last().unwrap())
    }

    fn open_drop_for_tuple(&mut self, tys: &[Ty<'tcx>]) -> BasicBlock {
        debug!("open_drop_for_tuple({:?}, {:?})", self, tys);

        let fields = tys
            .iter()
            .enumerate()
            .map(|(i, &ty)| {
                (
                    self.tcx().mk_place_field(self.place, FieldIdx::new(i), ty),
                    self.elaborator.field_subpath(self.path, FieldIdx::new(i)),
                )
            })
            .collect();

        let (succ, unwind) = self.drop_ladder_bottom();
        self.drop_ladder(fields, succ, unwind).0
    }

    #[instrument(level = "debug", ret)]
    fn open_drop_for_box(&mut self, adt: ty::AdtDef<'tcx>, substs: SubstsRef<'tcx>) -> BasicBlock {
        // drop glue is sent straight to codegen
        // box cannot be directly dereferenced
        let unique_ty = adt.non_enum_variant().fields[0].ty(self.tcx(), substs);
        let nonnull_ty =
            unique_ty.ty_adt_def().unwrap().non_enum_variant().fields[0].ty(self.tcx(), substs);
        let ptr_ty = self.tcx().mk_imm_ptr(substs[0].expect_ty());

        let unique_place = self.tcx().mk_place_field(self.place, FieldIdx::new(0), unique_ty);
        let nonnull_place = self.tcx().mk_place_field(unique_place, FieldIdx::new(0), nonnull_ty);
        let ptr_place = self.tcx().mk_place_field(nonnull_place, FieldIdx::new(0), ptr_ty);
        let interior = self.tcx().mk_place_deref(ptr_place);

        let interior_path = self.elaborator.deref_subpath(self.path);

        let succ = self.box_free_block(adt, substs, self.succ, self.unwind);
        let unwind_succ =
            self.unwind.map(|unwind| self.box_free_block(adt, substs, unwind, Unwind::InCleanup));

        self.drop_subpath(interior, interior_path, succ, unwind_succ)
    }

    #[instrument(level = "debug", ret)]
    fn open_drop_for_adt(&mut self, adt: ty::AdtDef<'tcx>, substs: SubstsRef<'tcx>) -> BasicBlock {
        if adt.variants().is_empty() {
            return self.elaborator.patch().new_block(BasicBlockData {
                statements: vec![],
                terminator: Some(Terminator {
                    source_info: self.source_info,
                    kind: TerminatorKind::Unreachable,
                }),
                is_cleanup: self.unwind.is_cleanup(),
            });
        }

        let skip_contents =
            adt.is_union() || Some(adt.did()) == self.tcx().lang_items().manually_drop();
        let contents_drop = if skip_contents {
            (self.succ, self.unwind)
        } else {
            self.open_drop_for_adt_contents(adt, substs)
        };

        if adt.has_dtor(self.tcx()) {
            self.destructor_call_block(contents_drop)
        } else {
            contents_drop.0
        }
    }

    fn open_drop_for_adt_contents(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        substs: SubstsRef<'tcx>,
    ) -> (BasicBlock, Unwind) {
        let (succ, unwind) = self.drop_ladder_bottom();
        if !adt.is_enum() {
            let fields = self.move_paths_for_fields(
                self.place,
                self.path,
                &adt.variant(FIRST_VARIANT),
                substs,
            );
            self.drop_ladder(fields, succ, unwind)
        } else {
            self.open_drop_for_multivariant(adt, substs, succ, unwind)
        }
    }

    fn open_drop_for_multivariant(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        substs: SubstsRef<'tcx>,
        succ: BasicBlock,
        unwind: Unwind,
    ) -> (BasicBlock, Unwind) {
        let mut values = Vec::with_capacity(adt.variants().len());
        let mut normal_blocks = Vec::with_capacity(adt.variants().len());
        let mut unwind_blocks =
            if unwind.is_cleanup() { None } else { Some(Vec::with_capacity(adt.variants().len())) };

        let mut have_otherwise_with_drop_glue = false;
        let mut have_otherwise = false;
        let tcx = self.tcx();

        for (variant_index, discr) in adt.discriminants(tcx) {
            let variant = &adt.variant(variant_index);
            let subpath = self.elaborator.downcast_subpath(self.path, variant_index);

            if let Some(variant_path) = subpath {
                let base_place = tcx.mk_place_elem(
                    self.place,
                    ProjectionElem::Downcast(Some(variant.name), variant_index),
                );
                let fields = self.move_paths_for_fields(base_place, variant_path, &variant, substs);
                values.push(discr.val);
                if let Unwind::To(unwind) = unwind {
                    // We can't use the half-ladder from the original
                    // drop ladder, because this breaks the
                    // "funclet can't have 2 successor funclets"
                    // requirement from MSVC:
                    //
                    //           switch       unwind-switch
                    //          /      \         /        \
                    //         v1.0    v2.0  v2.0-unwind  v1.0-unwind
                    //         |        |      /             |
                    //    v1.1-unwind  v2.1-unwind           |
                    //      ^                                |
                    //       \-------------------------------/
                    //
                    // Create a duplicate half-ladder to avoid that. We
                    // could technically only do this on MSVC, but I
                    // I want to minimize the divergence between MSVC
                    // and non-MSVC.

                    let unwind_blocks = unwind_blocks.as_mut().unwrap();
                    let unwind_ladder = vec![Unwind::InCleanup; fields.len() + 1];
                    let halfladder = self.drop_halfladder(&unwind_ladder, unwind, &fields);
                    unwind_blocks.push(halfladder.last().cloned().unwrap());
                }
                let (normal, _) = self.drop_ladder(fields, succ, unwind);
                normal_blocks.push(normal);
            } else {
                have_otherwise = true;

                let param_env = self.elaborator.param_env();
                let have_field_with_drop_glue = variant
                    .fields
                    .iter()
                    .any(|field| field.ty(tcx, substs).needs_drop(tcx, param_env));
                if have_field_with_drop_glue {
                    have_otherwise_with_drop_glue = true;
                }
            }
        }

        if !have_otherwise {
            values.pop();
        } else if !have_otherwise_with_drop_glue {
            normal_blocks.push(self.goto_block(succ, unwind));
            if let Unwind::To(unwind) = unwind {
                unwind_blocks.as_mut().unwrap().push(self.goto_block(unwind, Unwind::InCleanup));
            }
        } else {
            normal_blocks.push(self.drop_block(succ, unwind));
            if let Unwind::To(unwind) = unwind {
                unwind_blocks.as_mut().unwrap().push(self.drop_block(unwind, Unwind::InCleanup));
            }
        }

        (
            self.adt_switch_block(adt, normal_blocks, &values, succ, unwind),
            unwind.map(|unwind| {
                self.adt_switch_block(
                    adt,
                    unwind_blocks.unwrap(),
                    &values,
                    unwind,
                    Unwind::InCleanup,
                )
            }),
        )
    }

    fn adt_switch_block(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        blocks: Vec<BasicBlock>,
        values: &[u128],
        succ: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        // If there are multiple variants, then if something
        // is present within the enum the discriminant, tracked
        // by the rest path, must be initialized.
        //
        // Additionally, we do not want to switch on the
        // discriminant after it is free-ed, because that
        // way lies only trouble.
        let discr_ty = adt.repr().discr_type().to_ty(self.tcx());
        let discr = Place::from(self.new_temp(discr_ty));
        let discr_rv = Rvalue::Discriminant(self.place);
        let switch_block = BasicBlockData {
            statements: vec![self.assign(discr, discr_rv)],
            terminator: Some(Terminator {
                source_info: self.source_info,
                kind: TerminatorKind::SwitchInt {
                    discr: Operand::Move(discr),
                    targets: SwitchTargets::new(
                        values.iter().copied().zip(blocks.iter().copied()),
                        *blocks.last().unwrap(),
                    ),
                },
            }),
            is_cleanup: unwind.is_cleanup(),
        };
        let switch_block = self.elaborator.patch().new_block(switch_block);
        self.drop_flag_test_block(switch_block, succ, unwind)
    }

    fn destructor_call_block(&mut self, (succ, unwind): (BasicBlock, Unwind)) -> BasicBlock {
        debug!("destructor_call_block({:?}, {:?})", self, succ);
        let tcx = self.tcx();
        let drop_trait = tcx.require_lang_item(LangItem::Drop, None);
        let drop_fn = tcx.associated_item_def_ids(drop_trait)[0];
        let ty = self.place_ty(self.place);

        let ref_ty =
            tcx.mk_ref(tcx.lifetimes.re_erased, ty::TypeAndMut { ty, mutbl: hir::Mutability::Mut });
        let ref_place = self.new_temp(ref_ty);
        let unit_temp = Place::from(self.new_temp(tcx.mk_unit()));

        let result = BasicBlockData {
            statements: vec![self.assign(
                Place::from(ref_place),
                Rvalue::Ref(
                    tcx.lifetimes.re_erased,
                    BorrowKind::Mut { allow_two_phase_borrow: false },
                    self.place,
                ),
            )],
            terminator: Some(Terminator {
                kind: TerminatorKind::Call {
                    func: Operand::function_handle(
                        tcx,
                        drop_fn,
                        [ty.into()],
                        self.source_info.span,
                    ),
                    args: vec![Operand::Move(Place::from(ref_place))],
                    destination: unit_temp,
                    target: Some(succ),
                    cleanup: unwind.into_option(),
                    from_hir_call: true,
                    fn_span: self.source_info.span,
                },
                source_info: self.source_info,
            }),
            is_cleanup: unwind.is_cleanup(),
        };
        self.elaborator.patch().new_block(result)
    }

    /// Create a loop that drops an array:
    ///
    /// ```text
    /// loop-block:
    ///    can_go = cur == len
    ///    if can_go then succ else drop-block
    /// drop-block:
    ///    ptr = &raw mut P[cur]
    ///    cur = cur + 1
    ///    drop(ptr)
    /// ```
    fn drop_loop(
        &mut self,
        succ: BasicBlock,
        cur: Local,
        len: Local,
        ety: Ty<'tcx>,
        unwind: Unwind,
    ) -> BasicBlock {
        let copy = |place: Place<'tcx>| Operand::Copy(place);
        let move_ = |place: Place<'tcx>| Operand::Move(place);
        let tcx = self.tcx();

        let ptr_ty = tcx.mk_ptr(ty::TypeAndMut { ty: ety, mutbl: hir::Mutability::Mut });
        let ptr = Place::from(self.new_temp(ptr_ty));
        let can_go = Place::from(self.new_temp(tcx.types.bool));
        let one = self.constant_usize(1);

        let drop_block = BasicBlockData {
            statements: vec![
                self.assign(
                    ptr,
                    Rvalue::AddressOf(Mutability::Mut, tcx.mk_place_index(self.place, cur)),
                ),
                self.assign(
                    cur.into(),
                    Rvalue::BinaryOp(BinOp::Add, Box::new((move_(cur.into()), one))),
                ),
            ],
            is_cleanup: unwind.is_cleanup(),
            terminator: Some(Terminator {
                source_info: self.source_info,
                // this gets overwritten by drop elaboration.
                kind: TerminatorKind::Unreachable,
            }),
        };
        let drop_block = self.elaborator.patch().new_block(drop_block);

        let loop_block = BasicBlockData {
            statements: vec![self.assign(
                can_go,
                Rvalue::BinaryOp(BinOp::Eq, Box::new((copy(Place::from(cur)), copy(len.into())))),
            )],
            is_cleanup: unwind.is_cleanup(),
            terminator: Some(Terminator {
                source_info: self.source_info,
                kind: TerminatorKind::if_(move_(can_go), succ, drop_block),
            }),
        };
        let loop_block = self.elaborator.patch().new_block(loop_block);

        self.elaborator.patch().patch_terminator(
            drop_block,
            TerminatorKind::Drop {
                place: tcx.mk_place_deref(ptr),
                target: loop_block,
                unwind: unwind.into_option(),
            },
        );

        loop_block
    }

    fn open_drop_for_array(&mut self, ety: Ty<'tcx>, opt_size: Option<u64>) -> BasicBlock {
        debug!("open_drop_for_array({:?}, {:?})", ety, opt_size);
        let tcx = self.tcx();

        if let Some(size) = opt_size {
            let fields: Vec<(Place<'tcx>, Option<D::Path>)> = (0..size)
                .map(|i| {
                    (
                        tcx.mk_place_elem(
                            self.place,
                            ProjectionElem::ConstantIndex {
                                offset: i,
                                min_length: size,
                                from_end: false,
                            },
                        ),
                        self.elaborator.array_subpath(self.path, i, size),
                    )
                })
                .collect();

            if fields.iter().any(|(_, path)| path.is_some()) {
                let (succ, unwind) = self.drop_ladder_bottom();
                return self.drop_ladder(fields, succ, unwind).0;
            }
        }

        self.drop_loop_pair(ety)
    }

    /// Creates a pair of drop-loops of `place`, which drops its contents, even
    /// in the case of 1 panic.
    fn drop_loop_pair(&mut self, ety: Ty<'tcx>) -> BasicBlock {
        debug!("drop_loop_pair({:?})", ety);
        let tcx = self.tcx();
        let len = self.new_temp(tcx.types.usize);
        let cur = self.new_temp(tcx.types.usize);

        let unwind =
            self.unwind.map(|unwind| self.drop_loop(unwind, cur, len, ety, Unwind::InCleanup));

        let loop_block = self.drop_loop(self.succ, cur, len, ety, unwind);

        let zero = self.constant_usize(0);
        let block = BasicBlockData {
            statements: vec![
                self.assign(len.into(), Rvalue::Len(self.place)),
                self.assign(cur.into(), Rvalue::Use(zero)),
            ],
            is_cleanup: unwind.is_cleanup(),
            terminator: Some(Terminator {
                source_info: self.source_info,
                kind: TerminatorKind::Goto { target: loop_block },
            }),
        };

        let drop_block = self.elaborator.patch().new_block(block);
        // FIXME(#34708): handle partially-dropped array/slice elements.
        let reset_block = self.drop_flag_reset_block(DropFlagMode::Deep, drop_block, unwind);
        self.drop_flag_test_block(reset_block, self.succ, unwind)
    }

    /// The slow-path - create an "open", elaborated drop for a type
    /// which is moved-out-of only partially, and patch `bb` to a jump
    /// to it. This must not be called on ADTs with a destructor,
    /// as these can't be moved-out-of, except for `Box<T>`, which is
    /// special-cased.
    ///
    /// This creates a "drop ladder" that drops the needed fields of the
    /// ADT, both in the success case or if one of the destructors fail.
    fn open_drop(&mut self) -> BasicBlock {
        let ty = self.place_ty(self.place);
        match ty.kind() {
            ty::Closure(_, substs) => {
                let tys: Vec<_> = substs.as_closure().upvar_tys().collect();
                self.open_drop_for_tuple(&tys)
            }
            // Note that `elaborate_drops` only drops the upvars of a generator,
            // and this is ok because `open_drop` here can only be reached
            // within that own generator's resume function.
            // This should only happen for the self argument on the resume function.
            // It effectively only contains upvars until the generator transformation runs.
            // See librustc_body/transform/generator.rs for more details.
            ty::Generator(_, substs, _) => {
                let tys: Vec<_> = substs.as_generator().upvar_tys().collect();
                self.open_drop_for_tuple(&tys)
            }
            ty::Tuple(fields) => self.open_drop_for_tuple(fields),
            ty::Adt(def, substs) => {
                if def.is_box() {
                    self.open_drop_for_box(*def, substs)
                } else {
                    self.open_drop_for_adt(*def, substs)
                }
            }
            ty::Dynamic(..) => self.complete_drop(self.succ, self.unwind),
            ty::Array(ety, size) => {
                let size = size.try_eval_target_usize(self.tcx(), self.elaborator.param_env());
                self.open_drop_for_array(*ety, size)
            }
            ty::Slice(ety) => self.open_drop_for_array(*ety, None),

            _ => span_bug!(self.source_info.span, "open drop from non-ADT `{:?}`", ty),
        }
    }

    fn complete_drop(&mut self, succ: BasicBlock, unwind: Unwind) -> BasicBlock {
        debug!("complete_drop(succ={:?}, unwind={:?})", succ, unwind);

        let drop_block = self.drop_block(succ, unwind);

        self.drop_flag_test_block(drop_block, succ, unwind)
    }

    /// Creates a block that resets the drop flag. If `mode` is deep, all children drop flags will
    /// also be cleared.
    fn drop_flag_reset_block(
        &mut self,
        mode: DropFlagMode,
        succ: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        debug!("drop_flag_reset_block({:?},{:?})", self, mode);

        if unwind.is_cleanup() {
            // The drop flag isn't read again on the unwind path, so don't
            // bother setting it.
            return succ;
        }
        let block = self.new_block(unwind, TerminatorKind::Goto { target: succ });
        let block_start = Location { block, statement_index: 0 };
        self.elaborator.clear_drop_flag(block_start, self.path, mode);
        block
    }

    fn elaborated_drop_block(&mut self) -> BasicBlock {
        debug!("elaborated_drop_block({:?})", self);
        let blk = self.drop_block(self.succ, self.unwind);
        self.elaborate_drop(blk);
        blk
    }

    /// Creates a block that frees the backing memory of a `Box` if its drop is required (either
    /// statically or by checking its drop flag).
    ///
    /// The contained value will not be dropped.
    fn box_free_block(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        substs: SubstsRef<'tcx>,
        target: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        let block = self.unelaborated_free_block(adt, substs, target, unwind);
        self.drop_flag_test_block(block, target, unwind)
    }

    /// Creates a block that frees the backing memory of a `Box` (without dropping the contained
    /// value).
    fn unelaborated_free_block(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        substs: SubstsRef<'tcx>,
        target: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        let tcx = self.tcx();
        let unit_temp = Place::from(self.new_temp(tcx.mk_unit()));
        let free_func = tcx.require_lang_item(LangItem::BoxFree, Some(self.source_info.span));
        let args = adt
            .variant(FIRST_VARIANT)
            .fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let field = FieldIdx::new(i);
                let field_ty = f.ty(tcx, substs);
                Operand::Move(tcx.mk_place_field(self.place, field, field_ty))
            })
            .collect();

        let call = TerminatorKind::Call {
            func: Operand::function_handle(tcx, free_func, substs, self.source_info.span),
            args,
            destination: unit_temp,
            target: Some(target),
            cleanup: None,
            from_hir_call: false,
            fn_span: self.source_info.span,
        }; // FIXME(#43234)
        let free_block = self.new_block(unwind, call);

        let block_start = Location { block: free_block, statement_index: 0 };
        self.elaborator.clear_drop_flag(block_start, self.path, DropFlagMode::Shallow);
        free_block
    }

    fn drop_block(&mut self, target: BasicBlock, unwind: Unwind) -> BasicBlock {
        let block =
            TerminatorKind::Drop { place: self.place, target, unwind: unwind.into_option() };
        self.new_block(unwind, block)
    }

    fn goto_block(&mut self, target: BasicBlock, unwind: Unwind) -> BasicBlock {
        let block = TerminatorKind::Goto { target };
        self.new_block(unwind, block)
    }

    /// Returns the block to jump to in order to test the drop flag and execute the drop.
    ///
    /// Depending on the required `DropStyle`, this might be a generated block with an `if`
    /// terminator (for dynamic/open drops), or it might be `on_set` or `on_unset` itself, in case
    /// the drop can be statically determined.
    fn drop_flag_test_block(
        &mut self,
        on_set: BasicBlock,
        on_unset: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        let style = self.elaborator.drop_style(self.path, DropFlagMode::Shallow);
        debug!(
            "drop_flag_test_block({:?},{:?},{:?},{:?}) - {:?}",
            self, on_set, on_unset, unwind, style
        );

        match style {
            DropStyle::Dead => on_unset,
            DropStyle::Static => on_set,
            DropStyle::Conditional | DropStyle::Open => {
                let flag = self.elaborator.get_drop_flag(self.path).unwrap();
                let term = TerminatorKind::if_(flag, on_set, on_unset);
                self.new_block(unwind, term)
            }
        }
    }

    fn new_block(&mut self, unwind: Unwind, k: TerminatorKind<'tcx>) -> BasicBlock {
        self.elaborator.patch().new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator { source_info: self.source_info, kind: k }),
            is_cleanup: unwind.is_cleanup(),
        })
    }

    fn new_temp(&mut self, ty: Ty<'tcx>) -> Local {
        self.elaborator.patch().new_temp(ty, self.source_info.span)
    }

    fn constant_usize(&self, val: u16) -> Operand<'tcx> {
        Operand::Constant(Box::new(Constant {
            span: self.source_info.span,
            user_ty: None,
            literal: ConstantKind::from_usize(self.tcx(), val.into()),
        }))
    }

    fn assign(&self, lhs: Place<'tcx>, rhs: Rvalue<'tcx>) -> Statement<'tcx> {
        Statement {
            source_info: self.source_info,
            kind: StatementKind::Assign(Box::new((lhs, rhs))),
        }
    }
}
