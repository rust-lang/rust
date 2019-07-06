use std::fmt;
use rustc::hir;
use rustc::mir::*;
use rustc::middle::lang_items;
use rustc::traits::Reveal;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::layout::VariantIdx;
use rustc::ty::subst::SubstsRef;
use rustc::ty::util::IntTypeExt;
use rustc_data_structures::indexed_vec::Idx;
use crate::util::patch::MirPatch;

use std::convert::TryInto;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum DropFlagState {
    Present, // i.e., initialized
    Absent, // i.e., deinitialized or "moved"
}

impl DropFlagState {
    pub fn value(self) -> bool {
        match self {
            DropFlagState::Present => true,
            DropFlagState::Absent => false
        }
    }
}

#[derive(Debug)]
pub enum DropStyle {
    Dead,
    Static,
    Conditional,
    Open,
}

#[derive(Debug)]
pub enum DropFlagMode {
    Shallow,
    Deep
}

#[derive(Copy, Clone, Debug)]
pub enum Unwind {
    To(BasicBlock),
    InCleanup
}

impl Unwind {
    fn is_cleanup(self) -> bool {
        match self {
            Unwind::To(..) => false,
            Unwind::InCleanup => true
        }
    }

    fn into_option(self) -> Option<BasicBlock> {
        match self {
            Unwind::To(bb) => Some(bb),
            Unwind::InCleanup => None,
        }
    }

    fn map<F>(self, f: F) -> Self where F: FnOnce(BasicBlock) -> BasicBlock {
        match self {
            Unwind::To(bb) => Unwind::To(f(bb)),
            Unwind::InCleanup => Unwind::InCleanup
        }
    }
}

pub trait DropElaborator<'a, 'tcx>: fmt::Debug {
    type Path : Copy + fmt::Debug;

    fn patch(&mut self) -> &mut MirPatch<'tcx>;
    fn body(&self) -> &'a Body<'tcx>;
    fn tcx(&self) -> TyCtxt<'tcx>;
    fn param_env(&self) -> ty::ParamEnv<'tcx>;

    fn drop_style(&self, path: Self::Path, mode: DropFlagMode) -> DropStyle;
    fn get_drop_flag(&mut self, path: Self::Path) -> Option<Operand<'tcx>>;
    fn clear_drop_flag(&mut self, location: Location, path: Self::Path, mode: DropFlagMode);


    fn field_subpath(&self, path: Self::Path, field: Field) -> Option<Self::Path>;
    fn deref_subpath(&self, path: Self::Path) -> Option<Self::Path>;
    fn downcast_subpath(&self, path: Self::Path, variant: VariantIdx) -> Option<Self::Path>;
    fn array_subpath(&self, path: Self::Path, index: u32, size: u32) -> Option<Self::Path>;
}

#[derive(Debug)]
struct DropCtxt<'l, 'b, 'tcx, D>
where
    D: DropElaborator<'b, 'tcx>,
{
    elaborator: &'l mut D,

    source_info: SourceInfo,

    place: &'l Place<'tcx>,
    path: D::Path,
    succ: BasicBlock,
    unwind: Unwind,
}

pub fn elaborate_drop<'b, 'tcx, D>(
    elaborator: &mut D,
    source_info: SourceInfo,
    place: &Place<'tcx>,
    path: D::Path,
    succ: BasicBlock,
    unwind: Unwind,
    bb: BasicBlock,
) where
    D: DropElaborator<'b, 'tcx>,
    'tcx: 'b,
{
    DropCtxt {
        elaborator, source_info, place, path, succ, unwind
    }.elaborate_drop(bb)
}

impl<'l, 'b, 'tcx, D> DropCtxt<'l, 'b, 'tcx, D>
where
    D: DropElaborator<'b, 'tcx>,
    'tcx: 'b,
{
    fn place_ty(&self, place: &Place<'tcx>) -> Ty<'tcx> {
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
        debug!("elaborate_drop({:?})", self);
        let style = self.elaborator.drop_style(self.path, DropFlagMode::Deep);
        debug!("elaborate_drop({:?}): live - {:?}", self, style);
        match style {
            DropStyle::Dead => {
                self.elaborator.patch().patch_terminator(bb, TerminatorKind::Goto {
                    target: self.succ
                });
            }
            DropStyle::Static => {
                let loc = self.terminator_loc(bb);
                self.elaborator.clear_drop_flag(loc, self.path, DropFlagMode::Deep);
                self.elaborator.patch().patch_terminator(bb, TerminatorKind::Drop {
                    location: self.place.clone(),
                    target: self.succ,
                    unwind: self.unwind.into_option(),
                });
            }
            DropStyle::Conditional => {
                let unwind = self.unwind; // FIXME(#43234)
                let succ = self.succ;
                let drop_bb = self.complete_drop(Some(DropFlagMode::Deep), succ, unwind);
                self.elaborator.patch().patch_terminator(bb, TerminatorKind::Goto {
                    target: drop_bb
                });
            }
            DropStyle::Open => {
                let drop_bb = self.open_drop();
                self.elaborator.patch().patch_terminator(bb, TerminatorKind::Goto {
                    target: drop_bb
                });
            }
        }
    }

    /// Returns the place and move path for each field of `variant`,
    /// (the move path is `None` if the field is a rest field).
    fn move_paths_for_fields(&self,
                             base_place: &Place<'tcx>,
                             variant_path: D::Path,
                             variant: &'tcx ty::VariantDef,
                             substs: SubstsRef<'tcx>)
                             -> Vec<(Place<'tcx>, Option<D::Path>)>
    {
        variant.fields.iter().enumerate().map(|(i, f)| {
            let field = Field::new(i);
            let subpath = self.elaborator.field_subpath(variant_path, field);

            assert_eq!(self.elaborator.param_env().reveal, Reveal::All);
            let field_ty = self.tcx().normalize_erasing_regions(
                self.elaborator.param_env(),
                f.ty(self.tcx(), substs),
            );
            (base_place.clone().field(field, field_ty), subpath)
        }).collect()
    }

    fn drop_subpath(&mut self,
                    place: &Place<'tcx>,
                    path: Option<D::Path>,
                    succ: BasicBlock,
                    unwind: Unwind)
                    -> BasicBlock
    {
        if let Some(path) = path {
            debug!("drop_subpath: for std field {:?}", place);

            DropCtxt {
                elaborator: self.elaborator,
                source_info: self.source_info,
                path, place, succ, unwind,
            }.elaborated_drop_block()
        } else {
            debug!("drop_subpath: for rest field {:?}", place);

            DropCtxt {
                elaborator: self.elaborator,
                source_info: self.source_info,
                place, succ, unwind,
                // Using `self.path` here to condition the drop on
                // our own drop flag.
                path: self.path
            }.complete_drop(None, succ, unwind)
        }
    }

    /// Creates one-half of the drop ladder for a list of fields, and return
    /// the list of steps in it in reverse order, with the first step
    /// dropping 0 fields and so on.
    ///
    /// `unwind_ladder` is such a list of steps in reverse order,
    /// which is called if the matching step of the drop glue panics.
    fn drop_halfladder(&mut self,
                       unwind_ladder: &[Unwind],
                       mut succ: BasicBlock,
                       fields: &[(Place<'tcx>, Option<D::Path>)])
                       -> Vec<BasicBlock>
    {
        Some(succ).into_iter().chain(
            fields.iter().rev().zip(unwind_ladder)
                .map(|(&(ref place, path), &unwind_succ)| {
                    succ = self.drop_subpath(place, path, succ, unwind_succ);
                    succ
                })
        ).collect()
    }

    fn drop_ladder_bottom(&mut self) -> (BasicBlock, Unwind) {
        // Clear the "master" drop flag at the end. This is needed
        // because the "master" drop protects the ADT's discriminant,
        // which is invalidated after the ADT is dropped.
        let (succ, unwind) = (self.succ, self.unwind); // FIXME(#43234)
        (
            self.drop_flag_reset_block(DropFlagMode::Shallow, succ, unwind),
            unwind.map(|unwind| {
                self.drop_flag_reset_block(DropFlagMode::Shallow, unwind, Unwind::InCleanup)
            })
        )
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
        fields.retain(|&(ref place, _)| {
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

        let normal_ladder =
            self.drop_halfladder(&unwind_ladder, succ, &fields);

        (*normal_ladder.last().unwrap(), *unwind_ladder.last().unwrap())
    }

    fn open_drop_for_tuple(&mut self, tys: &[Ty<'tcx>]) -> BasicBlock {
        debug!("open_drop_for_tuple({:?}, {:?})", self, tys);

        let fields = tys.iter().enumerate().map(|(i, &ty)| {
            (self.place.clone().field(Field::new(i), ty),
             self.elaborator.field_subpath(self.path, Field::new(i)))
        }).collect();

        let (succ, unwind) = self.drop_ladder_bottom();
        self.drop_ladder(fields, succ, unwind).0
    }

    fn open_drop_for_box(&mut self, adt: &'tcx ty::AdtDef, substs: SubstsRef<'tcx>) -> BasicBlock {
        debug!("open_drop_for_box({:?}, {:?}, {:?})", self, adt, substs);

        let interior = self.place.clone().deref();
        let interior_path = self.elaborator.deref_subpath(self.path);

        let succ = self.succ; // FIXME(#43234)
        let unwind = self.unwind;
        let succ = self.box_free_block(adt, substs, succ, unwind);
        let unwind_succ = self.unwind.map(|unwind| {
            self.box_free_block(adt, substs, unwind, Unwind::InCleanup)
        });

        self.drop_subpath(&interior, interior_path, succ, unwind_succ)
    }

    fn open_drop_for_adt(&mut self, adt: &'tcx ty::AdtDef, substs: SubstsRef<'tcx>) -> BasicBlock {
        debug!("open_drop_for_adt({:?}, {:?}, {:?})", self, adt, substs);
        if adt.variants.len() == 0 {
            return self.elaborator.patch().new_block(BasicBlockData {
                statements: vec![],
                terminator: Some(Terminator {
                    source_info: self.source_info,
                    kind: TerminatorKind::Unreachable
                }),
                is_cleanup: self.unwind.is_cleanup()
            });
        }

        let skip_contents =
            adt.is_union() || Some(adt.did) == self.tcx().lang_items().manually_drop();
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

    fn open_drop_for_adt_contents(&mut self, adt: &'tcx ty::AdtDef,
                                  substs: SubstsRef<'tcx>)
                                  -> (BasicBlock, Unwind) {
        let (succ, unwind) = self.drop_ladder_bottom();
        if !adt.is_enum() {
            let fields = self.move_paths_for_fields(
                self.place,
                self.path,
                &adt.variants[VariantIdx::new(0)],
                substs
            );
            self.drop_ladder(fields, succ, unwind)
        } else {
            self.open_drop_for_multivariant(adt, substs, succ, unwind)
        }
    }

    fn open_drop_for_multivariant(&mut self, adt: &'tcx ty::AdtDef,
                                  substs: SubstsRef<'tcx>,
                                  succ: BasicBlock,
                                  unwind: Unwind)
                                  -> (BasicBlock, Unwind) {
        let mut values = Vec::with_capacity(adt.variants.len());
        let mut normal_blocks = Vec::with_capacity(adt.variants.len());
        let mut unwind_blocks = if unwind.is_cleanup() {
            None
        } else {
            Some(Vec::with_capacity(adt.variants.len()))
        };

        let mut have_otherwise = false;

        for (variant_index, discr) in adt.discriminants(self.tcx()) {
            let subpath = self.elaborator.downcast_subpath(
                self.path, variant_index);
            if let Some(variant_path) = subpath {
                let base_place = self.place.clone().elem(
                    ProjectionElem::Downcast(Some(adt.variants[variant_index].ident.name),
                                             variant_index));
                let fields = self.move_paths_for_fields(
                    &base_place,
                    variant_path,
                    &adt.variants[variant_index],
                    substs);
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
                    let halfladder =
                        self.drop_halfladder(&unwind_ladder, unwind, &fields);
                    unwind_blocks.push(halfladder.last().cloned().unwrap());
                }
                let (normal, _) = self.drop_ladder(fields, succ, unwind);
                normal_blocks.push(normal);
            } else {
                have_otherwise = true;
            }
        }

        if have_otherwise {
            normal_blocks.push(self.drop_block(succ, unwind));
            if let Unwind::To(unwind) = unwind {
                unwind_blocks.as_mut().unwrap().push(
                    self.drop_block(unwind, Unwind::InCleanup)
                        );
            }
        } else {
            values.pop();
        }

        (self.adt_switch_block(adt, normal_blocks, &values, succ, unwind),
         unwind.map(|unwind| {
             self.adt_switch_block(
                 adt, unwind_blocks.unwrap(), &values, unwind, Unwind::InCleanup
             )
         }))
    }

    fn adt_switch_block(&mut self,
                        adt: &'tcx ty::AdtDef,
                        blocks: Vec<BasicBlock>,
                        values: &[u128],
                        succ: BasicBlock,
                        unwind: Unwind)
                        -> BasicBlock {
        // If there are multiple variants, then if something
        // is present within the enum the discriminant, tracked
        // by the rest path, must be initialized.
        //
        // Additionally, we do not want to switch on the
        // discriminant after it is free-ed, because that
        // way lies only trouble.
        let discr_ty = adt.repr.discr_type().to_ty(self.tcx());
        let discr = Place::from(self.new_temp(discr_ty));
        let discr_rv = Rvalue::Discriminant(self.place.clone());
        let switch_block = BasicBlockData {
            statements: vec![self.assign(&discr, discr_rv)],
            terminator: Some(Terminator {
                source_info: self.source_info,
                kind: TerminatorKind::SwitchInt {
                    discr: Operand::Move(discr),
                    switch_ty: discr_ty,
                    values: From::from(values.to_owned()),
                    targets: blocks,
                }
            }),
            is_cleanup: unwind.is_cleanup(),
        };
        let switch_block = self.elaborator.patch().new_block(switch_block);
        self.drop_flag_test_block(switch_block, succ, unwind)
    }

    fn destructor_call_block(&mut self, (succ, unwind): (BasicBlock, Unwind)) -> BasicBlock {
        debug!("destructor_call_block({:?}, {:?})", self, succ);
        let tcx = self.tcx();
        let drop_trait = tcx.lang_items().drop_trait().unwrap();
        let drop_fn = tcx.associated_items(drop_trait).next().unwrap();
        let ty = self.place_ty(self.place);
        let substs = tcx.mk_substs_trait(ty, &[]);

        let ref_ty = tcx.mk_ref(tcx.lifetimes.re_erased, ty::TypeAndMut {
            ty,
            mutbl: hir::Mutability::MutMutable
        });
        let ref_place = self.new_temp(ref_ty);
        let unit_temp = Place::from(self.new_temp(tcx.mk_unit()));

        let result = BasicBlockData {
            statements: vec![self.assign(
                &Place::from(ref_place),
                Rvalue::Ref(tcx.lifetimes.re_erased,
                            BorrowKind::Mut { allow_two_phase_borrow: false },
                            self.place.clone())
            )],
            terminator: Some(Terminator {
                kind: TerminatorKind::Call {
                    func: Operand::function_handle(tcx, drop_fn.def_id, substs,
                                                   self.source_info.span),
                    args: vec![Operand::Move(Place::from(ref_place))],
                    destination: Some((unit_temp, succ)),
                    cleanup: unwind.into_option(),
                    from_hir_call: true,
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
    ///    can_go = cur == length_or_end
    ///    if can_go then succ else drop-block
    /// drop-block:
    ///    if ptr_based {
    ///        ptr = &mut *cur
    ///        cur = cur.offset(1)
    ///    } else {
    ///        ptr = &mut P[cur]
    ///        cur = cur + 1
    ///    }
    ///    drop(ptr)
    /// ```
    fn drop_loop(
        &mut self,
        succ: BasicBlock,
        cur: Local,
        length_or_end: &Place<'tcx>,
        ety: Ty<'tcx>,
        unwind: Unwind,
        ptr_based: bool,
    ) -> BasicBlock {
        let copy = |place: &Place<'tcx>| Operand::Copy(place.clone());
        let move_ = |place: &Place<'tcx>| Operand::Move(place.clone());
        let tcx = self.tcx();

        let ref_ty = tcx.mk_ref(tcx.lifetimes.re_erased, ty::TypeAndMut {
            ty: ety,
            mutbl: hir::Mutability::MutMutable
        });
        let ptr = &Place::from(self.new_temp(ref_ty));
        let can_go = &Place::from(self.new_temp(tcx.types.bool));

        let one = self.constant_usize(1);
        let (ptr_next, cur_next) = if ptr_based {
            (Rvalue::Ref(
                tcx.lifetimes.re_erased,
                BorrowKind::Mut { allow_two_phase_borrow: false },
                Place::Projection(Box::new(Projection {
                    base: Place::Base(PlaceBase::Local(cur)),
                    elem: ProjectionElem::Deref,
                }))
             ),
             Rvalue::BinaryOp(BinOp::Offset, move_(&Place::from(cur)), one))
        } else {
            (Rvalue::Ref(
                 tcx.lifetimes.re_erased,
                 BorrowKind::Mut { allow_two_phase_borrow: false },
                 self.place.clone().index(cur)),
             Rvalue::BinaryOp(BinOp::Add, move_(&Place::from(cur)), one))
        };

        let drop_block = BasicBlockData {
            statements: vec![
                self.assign(ptr, ptr_next),
                self.assign(&Place::from(cur), cur_next)
            ],
            is_cleanup: unwind.is_cleanup(),
            terminator: Some(Terminator {
                source_info: self.source_info,
                // this gets overwritten by drop elaboration.
                kind: TerminatorKind::Unreachable,
            })
        };
        let drop_block = self.elaborator.patch().new_block(drop_block);

        let loop_block = BasicBlockData {
            statements: vec![
                self.assign(can_go, Rvalue::BinaryOp(BinOp::Eq,
                                                     copy(&Place::from(cur)),
                                                     copy(length_or_end)))
            ],
            is_cleanup: unwind.is_cleanup(),
            terminator: Some(Terminator {
                source_info: self.source_info,
                kind: TerminatorKind::if_(tcx, move_(can_go), succ, drop_block)
            })
        };
        let loop_block = self.elaborator.patch().new_block(loop_block);

        self.elaborator.patch().patch_terminator(drop_block, TerminatorKind::Drop {
            location: ptr.clone().deref(),
            target: loop_block,
            unwind: unwind.into_option()
        });

        loop_block
    }

    fn open_drop_for_array(&mut self, ety: Ty<'tcx>, opt_size: Option<u64>) -> BasicBlock {
        debug!("open_drop_for_array({:?}, {:?})", ety, opt_size);

        // if size_of::<ety>() == 0 {
        //     index_based_loop
        // } else {
        //     ptr_based_loop
        // }

        if let Some(size) = opt_size {
            let size: u32 = size.try_into().unwrap_or_else(|_| {
                bug!("move out check isn't implemented for array sizes bigger than u32::MAX");
            });
            let fields: Vec<(Place<'tcx>, Option<D::Path>)> = (0..size).map(|i| {
                (self.place.clone().elem(ProjectionElem::ConstantIndex{
                    offset: i,
                    min_length: size,
                    from_end: false
                }),
                 self.elaborator.array_subpath(self.path, i, size))
            }).collect();

            if fields.iter().any(|(_,path)| path.is_some()) {
                let (succ, unwind) = self.drop_ladder_bottom();
                return self.drop_ladder(fields, succ, unwind).0
            }
        }

        let move_ = |place: &Place<'tcx>| Operand::Move(place.clone());
        let tcx = self.tcx();
        let elem_size = &Place::from(self.new_temp(tcx.types.usize));
        let len = &Place::from(self.new_temp(tcx.types.usize));

        static USIZE_SWITCH_ZERO: &[u128] = &[0];

        let base_block = BasicBlockData {
            statements: vec![
                self.assign(elem_size, Rvalue::NullaryOp(NullOp::SizeOf, ety)),
                self.assign(len, Rvalue::Len(self.place.clone())),
            ],
            is_cleanup: self.unwind.is_cleanup(),
            terminator: Some(Terminator {
                source_info: self.source_info,
                kind: TerminatorKind::SwitchInt {
                    discr: move_(elem_size),
                    switch_ty: tcx.types.usize,
                    values: From::from(USIZE_SWITCH_ZERO),
                    targets: vec![
                        self.drop_loop_pair(ety, false, len.clone()),
                        self.drop_loop_pair(ety, true, len.clone()),
                    ],
                },
            })
        };
        self.elaborator.patch().new_block(base_block)
    }

    /// Ceates a pair of drop-loops of `place`, which drops its contents, even
    /// in the case of 1 panic. If `ptr_based`, creates a pointer loop,
    /// otherwise create an index loop.
    fn drop_loop_pair(
        &mut self,
        ety: Ty<'tcx>,
        ptr_based: bool,
        length: Place<'tcx>,
    ) -> BasicBlock {
        debug!("drop_loop_pair({:?}, {:?})", ety, ptr_based);
        let tcx = self.tcx();
        let iter_ty = if ptr_based {
            tcx.mk_mut_ptr(ety)
        } else {
            tcx.types.usize
        };

        let cur = self.new_temp(iter_ty);
        let length_or_end = if ptr_based {
            // FIXME check if we want to make it return a `Place` directly
            // if all use sites want a `Place::Base` anyway.
            Place::from(self.new_temp(iter_ty))
        } else {
            length.clone()
        };

        let unwind = self.unwind.map(|unwind| {
            self.drop_loop(unwind,
                           cur,
                           &length_or_end,
                           ety,
                           Unwind::InCleanup,
                           ptr_based)
        });

        let loop_block = self.drop_loop(
            self.succ,
            cur,
            &length_or_end,
            ety,
            unwind,
            ptr_based);

        let cur = Place::from(cur);
        let drop_block_stmts = if ptr_based {
            let tmp_ty = tcx.mk_mut_ptr(self.place_ty(self.place));
            let tmp = Place::from(self.new_temp(tmp_ty));
            // tmp = &mut P;
            // cur = tmp as *mut T;
            // end = Offset(cur, len);
            vec![
                self.assign(&tmp, Rvalue::Ref(
                    tcx.lifetimes.re_erased,
                    BorrowKind::Mut { allow_two_phase_borrow: false },
                    self.place.clone()
                )),
                self.assign(
                    &cur,
                    Rvalue::Cast(CastKind::Misc, Operand::Move(tmp), iter_ty),
                ),
                self.assign(
                    &length_or_end,
                    Rvalue::BinaryOp(BinOp::Offset, Operand::Copy(cur), Operand::Move(length)
                )),
            ]
        } else {
            // cur = 0 (length already pushed)
            let zero = self.constant_usize(0);
            vec![self.assign(&cur, Rvalue::Use(zero))]
        };
        let drop_block = self.elaborator.patch().new_block(BasicBlockData {
            statements: drop_block_stmts,
            is_cleanup: unwind.is_cleanup(),
            terminator: Some(Terminator {
                source_info: self.source_info,
                kind: TerminatorKind::Goto { target: loop_block }
            })
        });

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
        match ty.sty {
            ty::Closure(def_id, substs) => {
                let tys : Vec<_> = substs.upvar_tys(def_id, self.tcx()).collect();
                self.open_drop_for_tuple(&tys)
            }
            // Note that `elaborate_drops` only drops the upvars of a generator,
            // and this is ok because `open_drop` here can only be reached
            // within that own generator's resume function.
            // This should only happen for the self argument on the resume function.
            // It effetively only contains upvars until the generator transformation runs.
            // See librustc_body/transform/generator.rs for more details.
            ty::Generator(def_id, substs, _) => {
                let tys : Vec<_> = substs.upvar_tys(def_id, self.tcx()).collect();
                self.open_drop_for_tuple(&tys)
            }
            ty::Tuple(tys) => {
                let tys: Vec<_> = tys.iter().map(|k| k.expect_ty()).collect();
                self.open_drop_for_tuple(&tys)
            }
            ty::Adt(def, substs) => {
                if def.is_box() {
                    self.open_drop_for_box(def, substs)
                } else {
                    self.open_drop_for_adt(def, substs)
                }
            }
            ty::Dynamic(..) => {
                let unwind = self.unwind; // FIXME(#43234)
                let succ = self.succ;
                self.complete_drop(Some(DropFlagMode::Deep), succ, unwind)
            }
            ty::Array(ety, size) => {
                let size = size.assert_usize(self.tcx());
                self.open_drop_for_array(ety, size)
            },
            ty::Slice(ety) => self.open_drop_for_array(ety, None),

            _ => bug!("open drop from non-ADT `{:?}`", ty)
        }
    }

    /// Returns a basic block that drop a place using the context
    /// and path in `c`. If `mode` is something, also clear `c`
    /// according to it.
    ///
    /// if FLAG(self.path)
    ///     if let Some(mode) = mode: FLAG(self.path)[mode] = false
    ///     drop(self.place)
    fn complete_drop(
        &mut self,
        drop_mode: Option<DropFlagMode>,
        succ: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        debug!("complete_drop({:?},{:?})", self, drop_mode);

        let drop_block = self.drop_block(succ, unwind);
        let drop_block = if let Some(mode) = drop_mode {
            self.drop_flag_reset_block(mode, drop_block, unwind)
        } else {
            drop_block
        };

        self.drop_flag_test_block(drop_block, succ, unwind)
    }

    fn drop_flag_reset_block(&mut self,
                             mode: DropFlagMode,
                             succ: BasicBlock,
                             unwind: Unwind) -> BasicBlock
    {
        debug!("drop_flag_reset_block({:?},{:?})", self, mode);

        let block = self.new_block(unwind, TerminatorKind::Goto { target: succ });
        let block_start = Location { block: block, statement_index: 0 };
        self.elaborator.clear_drop_flag(block_start, self.path, mode);
        block
    }

    fn elaborated_drop_block(&mut self) -> BasicBlock {
        debug!("elaborated_drop_block({:?})", self);
        let unwind = self.unwind; // FIXME(#43234)
        let succ = self.succ;
        let blk = self.drop_block(succ, unwind);
        self.elaborate_drop(blk);
        blk
    }

    fn box_free_block(
        &mut self,
        adt: &'tcx ty::AdtDef,
        substs: SubstsRef<'tcx>,
        target: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        let block = self.unelaborated_free_block(adt, substs, target, unwind);
        self.drop_flag_test_block(block, target, unwind)
    }

    fn unelaborated_free_block(
        &mut self,
        adt: &'tcx ty::AdtDef,
        substs: SubstsRef<'tcx>,
        target: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        let tcx = self.tcx();
        let unit_temp = Place::from(self.new_temp(tcx.mk_unit()));
        let free_func = tcx.require_lang_item(lang_items::BoxFreeFnLangItem);
        let args = adt.variants[VariantIdx::new(0)].fields.iter().enumerate().map(|(i, f)| {
            let field = Field::new(i);
            let field_ty = f.ty(self.tcx(), substs);
            Operand::Move(self.place.clone().field(field, field_ty))
        }).collect();

        let call = TerminatorKind::Call {
            func: Operand::function_handle(tcx, free_func, substs, self.source_info.span),
            args: args,
            destination: Some((unit_temp, target)),
            cleanup: None,
            from_hir_call: false,
        }; // FIXME(#43234)
        let free_block = self.new_block(unwind, call);

        let block_start = Location { block: free_block, statement_index: 0 };
        self.elaborator.clear_drop_flag(block_start, self.path, DropFlagMode::Shallow);
        free_block
    }

    fn drop_block(&mut self, target: BasicBlock, unwind: Unwind) -> BasicBlock {
        let block = TerminatorKind::Drop {
            location: self.place.clone(),
            target,
            unwind: unwind.into_option()
        };
        self.new_block(unwind, block)
    }

    fn drop_flag_test_block(&mut self,
                            on_set: BasicBlock,
                            on_unset: BasicBlock,
                            unwind: Unwind)
                            -> BasicBlock
    {
        let style = self.elaborator.drop_style(self.path, DropFlagMode::Shallow);
        debug!("drop_flag_test_block({:?},{:?},{:?},{:?}) - {:?}",
               self, on_set, on_unset, unwind, style);

        match style {
            DropStyle::Dead => on_unset,
            DropStyle::Static => on_set,
            DropStyle::Conditional | DropStyle::Open => {
                let flag = self.elaborator.get_drop_flag(self.path).unwrap();
                let term = TerminatorKind::if_(self.tcx(), flag, on_set, on_unset);
                self.new_block(unwind, term)
            }
        }
    }

    fn new_block(&mut self, unwind: Unwind, k: TerminatorKind<'tcx>) -> BasicBlock {
        self.elaborator.patch().new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: self.source_info, kind: k
            }),
            is_cleanup: unwind.is_cleanup()
        })
    }

    fn new_temp(&mut self, ty: Ty<'tcx>) -> Local {
        self.elaborator.patch().new_temp(ty, self.source_info.span)
    }

    fn terminator_loc(&mut self, bb: BasicBlock) -> Location {
        let body = self.elaborator.body();
        self.elaborator.patch().terminator_loc(body, bb)
    }

    fn constant_usize(&self, val: u16) -> Operand<'tcx> {
        Operand::Constant(box Constant {
            span: self.source_info.span,
            ty: self.tcx().types.usize,
            user_ty: None,
            literal: ty::Const::from_usize(self.tcx(), val.into()),
        })
    }

    fn assign(&self, lhs: &Place<'tcx>, rhs: Rvalue<'tcx>) -> Statement<'tcx> {
        Statement {
            source_info: self.source_info,
            kind: StatementKind::Assign(lhs.clone(), box rhs)
        }
    }
}
