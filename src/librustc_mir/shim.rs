use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::mir::*;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::layout::VariantIdx;
use rustc::ty::subst::{Subst, InternalSubsts};
use rustc::ty::query::Providers;

use rustc_data_structures::indexed_vec::{IndexVec, Idx};

use rustc_target::spec::abi::Abi;
use syntax_pos::{Span, sym};

use std::fmt;
use std::iter;

use crate::transform::{
    add_moves_for_packed_drops, add_call_guards,
    remove_noop_landing_pads, no_landing_pads, simplify, run_passes
};
use crate::util::elaborate_drops::{self, DropElaborator, DropStyle, DropFlagMode};
use crate::util::patch::MirPatch;
use crate::util::expand_aggregate;

pub fn provide(providers: &mut Providers<'_>) {
    providers.mir_shims = make_shim;
}

fn make_shim<'tcx>(tcx: TyCtxt<'tcx>, instance: ty::InstanceDef<'tcx>) -> &'tcx Body<'tcx> {
    debug!("make_shim({:?})", instance);

    let mut result = match instance {
        ty::InstanceDef::Item(..) =>
            bug!("item {:?} passed to make_shim", instance),
        ty::InstanceDef::VtableShim(def_id) => {
            build_call_shim(
                tcx,
                def_id,
                Adjustment::DerefMove,
                CallKind::Direct(def_id),
                None,
            )
        }
        ty::InstanceDef::FnPtrShim(def_id, ty) => {
            let trait_ = tcx.trait_of_item(def_id).unwrap();
            let adjustment = match tcx.lang_items().fn_trait_kind(trait_) {
                Some(ty::ClosureKind::FnOnce) => Adjustment::Identity,
                Some(ty::ClosureKind::FnMut) |
                Some(ty::ClosureKind::Fn) => Adjustment::Deref,
                None => bug!("fn pointer {:?} is not an fn", ty)
            };
            // HACK: we need the "real" argument types for the MIR,
            // but because our substs are (Self, Args), where Args
            // is a tuple, we must include the *concrete* argument
            // types in the MIR. They will be substituted again with
            // the param-substs, but because they are concrete, this
            // will not do any harm.
            let sig = tcx.erase_late_bound_regions(&ty.fn_sig(tcx));
            let arg_tys = sig.inputs();

            build_call_shim(
                tcx,
                def_id,
                adjustment,
                CallKind::Indirect,
                Some(arg_tys)
            )
        }
        ty::InstanceDef::Virtual(def_id, _) => {
            // We are generating a call back to our def-id, which the
            // codegen backend knows to turn to an actual virtual call.
            build_call_shim(
                tcx,
                def_id,
                Adjustment::Identity,
                CallKind::Direct(def_id),
                None
            )
        }
        ty::InstanceDef::ClosureOnceShim { call_once } => {
            let fn_mut = tcx.lang_items().fn_mut_trait().unwrap();
            let call_mut = tcx.global_tcx()
                .associated_items(fn_mut)
                .find(|it| it.kind == ty::AssocKind::Method)
                .unwrap().def_id;

            build_call_shim(
                tcx,
                call_once,
                Adjustment::RefMut,
                CallKind::Direct(call_mut),
                None
            )
        }
        ty::InstanceDef::DropGlue(def_id, ty) => {
            build_drop_shim(tcx, def_id, ty)
        }
        ty::InstanceDef::CloneShim(def_id, ty) => {
            let name = tcx.item_name(def_id);
            if name == sym::clone {
                build_clone_shim(tcx, def_id, ty)
            } else if name == sym::clone_from {
                debug!("make_shim({:?}: using default trait implementation", instance);
                return tcx.optimized_mir(def_id);
            } else {
                bug!("builtin clone shim {:?} not supported", instance)
            }
        }
        ty::InstanceDef::Intrinsic(_) => {
            bug!("creating shims from intrinsics ({:?}) is unsupported", instance)
        }
    };
    debug!("make_shim({:?}) = untransformed {:?}", instance, result);

    run_passes(tcx, &mut result, instance, MirPhase::Const, &[
        &add_moves_for_packed_drops::AddMovesForPackedDrops,
        &no_landing_pads::NoLandingPads,
        &remove_noop_landing_pads::RemoveNoopLandingPads,
        &simplify::SimplifyCfg::new("make_shim"),
        &add_call_guards::CriticalCallEdges,
    ]);

    debug!("make_shim({:?}) = {:?}", instance, result);

    tcx.arena.alloc(result)
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Adjustment {
    Identity,
    Deref,
    DerefMove,
    RefMut,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum CallKind {
    Indirect,
    Direct(DefId),
}

fn temp_decl(mutability: Mutability, ty: Ty<'_>, span: Span) -> LocalDecl<'_> {
    let source_info = SourceInfo { scope: OUTERMOST_SOURCE_SCOPE, span };
    LocalDecl {
        mutability,
        ty,
        user_ty: UserTypeProjections::none(),
        name: None,
        source_info,
        visibility_scope: source_info.scope,
        internal: false,
        is_user_variable: None,
        is_block_tail: None,
    }
}

fn local_decls_for_sig<'tcx>(sig: &ty::FnSig<'tcx>, span: Span)
    -> IndexVec<Local, LocalDecl<'tcx>>
{
    iter::once(temp_decl(Mutability::Mut, sig.output(), span))
        .chain(sig.inputs().iter().map(
            |ity| temp_decl(Mutability::Not, ity, span)))
        .collect()
}

fn build_drop_shim<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, ty: Option<Ty<'tcx>>) -> Body<'tcx> {
    debug!("build_drop_shim(def_id={:?}, ty={:?})", def_id, ty);

    // Check if this is a generator, if so, return the drop glue for it
    if let Some(&ty::TyS { sty: ty::Generator(gen_def_id, substs, _), .. }) = ty {
        let body = &**tcx.optimized_mir(gen_def_id).generator_drop.as_ref().unwrap();
        return body.subst(tcx, substs.substs);
    }

    let substs = if let Some(ty) = ty {
        tcx.intern_substs(&[ty.into()])
    } else {
        InternalSubsts::identity_for_item(tcx, def_id)
    };
    let sig = tcx.fn_sig(def_id).subst(tcx, substs);
    let sig = tcx.erase_late_bound_regions(&sig);
    let span = tcx.def_span(def_id);

    let source_info = SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE };

    let return_block = BasicBlock::new(1);
    let mut blocks = IndexVec::with_capacity(2);
    let block = |blocks: &mut IndexVec<_, _>, kind| {
        blocks.push(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup: false
        })
    };
    block(&mut blocks, TerminatorKind::Goto { target: return_block });
    block(&mut blocks, TerminatorKind::Return);

    let mut body = Body::new(
        blocks,
        IndexVec::from_elem_n(
            SourceScopeData { span: span, parent_scope: None }, 1
        ),
        ClearCrossCrate::Clear,
        IndexVec::new(),
        None,
        local_decls_for_sig(&sig, span),
        IndexVec::new(),
        sig.inputs().len(),
        vec![],
        span,
        vec![],
    );

    if let Some(..) = ty {
        // The first argument (index 0), but add 1 for the return value.
        let dropee_ptr = Place::from(Local::new(1+0));
        if tcx.sess.opts.debugging_opts.mir_emit_retag {
            // Function arguments should be retagged, and we make this one raw.
            body.basic_blocks_mut()[START_BLOCK].statements.insert(0, Statement {
                source_info,
                kind: StatementKind::Retag(RetagKind::Raw, dropee_ptr.clone()),
            });
        }
        let patch = {
            let param_env = tcx.param_env(def_id).with_reveal_all();
            let mut elaborator = DropShimElaborator {
                body: &body,
                patch: MirPatch::new(&body),
                tcx,
                param_env
            };
            let dropee = dropee_ptr.deref();
            let resume_block = elaborator.patch.resume_block();
            elaborate_drops::elaborate_drop(
                &mut elaborator,
                source_info,
                &dropee,
                (),
                return_block,
                elaborate_drops::Unwind::To(resume_block),
                START_BLOCK
            );
            elaborator.patch
        };
        patch.apply(&mut body);
    }

    body
}

pub struct DropShimElaborator<'a, 'tcx> {
    pub body: &'a Body<'tcx>,
    pub patch: MirPatch<'tcx>,
    pub tcx: TyCtxt<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
}

impl<'a, 'tcx> fmt::Debug for DropShimElaborator<'a, 'tcx> {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        Ok(())
    }
}

impl<'a, 'tcx> DropElaborator<'a, 'tcx> for DropShimElaborator<'a, 'tcx> {
    type Path = ();

    fn patch(&mut self) -> &mut MirPatch<'tcx> { &mut self.patch }
    fn body(&self) -> &'a Body<'tcx> { self.body }
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
        }
    fn param_env(&self) -> ty::ParamEnv<'tcx> { self.param_env }

    fn drop_style(&self, _path: Self::Path, mode: DropFlagMode) -> DropStyle {
        if let DropFlagMode::Shallow = mode {
            DropStyle::Static
        } else {
            DropStyle::Open
        }
    }

    fn get_drop_flag(&mut self, _path: Self::Path) -> Option<Operand<'tcx>> {
        None
    }

    fn clear_drop_flag(&mut self, _location: Location, _path: Self::Path, _mode: DropFlagMode) {
    }

    fn field_subpath(&self, _path: Self::Path, _field: Field) -> Option<Self::Path> {
        None
    }
    fn deref_subpath(&self, _path: Self::Path) -> Option<Self::Path> {
        None
    }
    fn downcast_subpath(&self, _path: Self::Path, _variant: VariantIdx) -> Option<Self::Path> {
        Some(())
    }
    fn array_subpath(&self, _path: Self::Path, _index: u32, _size: u32) -> Option<Self::Path> {
        None
    }
}

/// Builds a `Clone::clone` shim for `self_ty`. Here, `def_id` is `Clone::clone`.
fn build_clone_shim<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, self_ty: Ty<'tcx>) -> Body<'tcx> {
    debug!("build_clone_shim(def_id={:?})", def_id);

    let mut builder = CloneShimBuilder::new(tcx, def_id, self_ty);
    let is_copy = self_ty.is_copy_modulo_regions(tcx, tcx.param_env(def_id), builder.span);

    let dest = Place::RETURN_PLACE;
    let src = Place::from(Local::new(1+0)).deref();

    match self_ty.sty {
        _ if is_copy => builder.copy_shim(),
        ty::Array(ty, len) => {
            let len = len.unwrap_usize(tcx);
            builder.array_shim(dest, src, ty, len)
        }
        ty::Closure(def_id, substs) => {
            builder.tuple_like_shim(
                dest, src,
                substs.upvar_tys(def_id, tcx)
            )
        }
        ty::Tuple(tys) => builder.tuple_like_shim(dest, src, tys.iter().map(|k| k.expect_ty())),
        _ => {
            bug!("clone shim for `{:?}` which is not `Copy` and is not an aggregate", self_ty)
        }
    };

    builder.into_mir()
}

struct CloneShimBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
    blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    span: Span,
    sig: ty::FnSig<'tcx>,
}

impl CloneShimBuilder<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def_id: DefId, self_ty: Ty<'tcx>) -> Self {
        // we must subst the self_ty because it's
        // otherwise going to be TySelf and we can't index
        // or access fields of a Place of type TySelf.
        let substs = tcx.mk_substs_trait(self_ty, &[]);
        let sig = tcx.fn_sig(def_id).subst(tcx, substs);
        let sig = tcx.erase_late_bound_regions(&sig);
        let span = tcx.def_span(def_id);

        CloneShimBuilder {
            tcx,
            def_id,
            local_decls: local_decls_for_sig(&sig, span),
            blocks: IndexVec::new(),
            span,
            sig,
        }
    }

    fn into_mir(self) -> Body<'tcx> {
        Body::new(
            self.blocks,
            IndexVec::from_elem_n(
                SourceScopeData { span: self.span, parent_scope: None }, 1
            ),
            ClearCrossCrate::Clear,
            IndexVec::new(),
            None,
            self.local_decls,
            IndexVec::new(),
            self.sig.inputs().len(),
            vec![],
            self.span,
            vec![],
        )
    }

    fn source_info(&self) -> SourceInfo {
        SourceInfo { span: self.span, scope: OUTERMOST_SOURCE_SCOPE }
    }

    fn block(
        &mut self,
        statements: Vec<Statement<'tcx>>,
        kind: TerminatorKind<'tcx>,
        is_cleanup: bool
    ) -> BasicBlock {
        let source_info = self.source_info();
        self.blocks.push(BasicBlockData {
            statements,
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup,
        })
    }

    /// Gives the index of an upcoming BasicBlock, with an offset.
    /// offset=0 will give you the index of the next BasicBlock,
    /// offset=1 will give the index of the next-to-next block,
    /// offset=-1 will give you the index of the last-created block
    fn block_index_offset(&mut self, offset: usize) -> BasicBlock {
        BasicBlock::new(self.blocks.len() + offset)
    }

    fn make_statement(&self, kind: StatementKind<'tcx>) -> Statement<'tcx> {
        Statement {
            source_info: self.source_info(),
            kind,
        }
    }

    fn copy_shim(&mut self) {
        let rcvr = Place::from(Local::new(1+0)).deref();
        let ret_statement = self.make_statement(
            StatementKind::Assign(
                Place::RETURN_PLACE,
                box Rvalue::Use(Operand::Copy(rcvr))
            )
        );
        self.block(vec![ret_statement], TerminatorKind::Return, false);
    }

    fn make_place(&mut self, mutability: Mutability, ty: Ty<'tcx>) -> Place<'tcx> {
        let span = self.span;
        Place::from(self.local_decls.push(temp_decl(mutability, ty, span)))
    }

    fn make_clone_call(
        &mut self,
        dest: Place<'tcx>,
        src: Place<'tcx>,
        ty: Ty<'tcx>,
        next: BasicBlock,
        cleanup: BasicBlock
    ) {
        let tcx = self.tcx;

        let substs = tcx.mk_substs_trait(ty, &[]);

        // `func == Clone::clone(&ty) -> ty`
        let func_ty = tcx.mk_fn_def(self.def_id, substs);
        let func = Operand::Constant(box Constant {
            span: self.span,
            ty: func_ty,
            user_ty: None,
            literal: ty::Const::zero_sized(tcx, func_ty),
        });

        let ref_loc = self.make_place(
            Mutability::Not,
            tcx.mk_ref(tcx.lifetimes.re_erased, ty::TypeAndMut {
                ty,
                mutbl: hir::Mutability::MutImmutable,
            })
        );

        // `let ref_loc: &ty = &src;`
        let statement = self.make_statement(
            StatementKind::Assign(
                ref_loc.clone(),
                box Rvalue::Ref(tcx.lifetimes.re_erased, BorrowKind::Shared, src)
            )
        );

        // `let loc = Clone::clone(ref_loc);`
        self.block(vec![statement], TerminatorKind::Call {
            func,
            args: vec![Operand::Move(ref_loc)],
            destination: Some((dest, next)),
            cleanup: Some(cleanup),
            from_hir_call: true,
        }, false);
    }

    fn loop_header(
        &mut self,
        beg: Place<'tcx>,
        end: Place<'tcx>,
        loop_body: BasicBlock,
        loop_end: BasicBlock,
        is_cleanup: bool
    ) {
        let tcx = self.tcx;

        let cond = self.make_place(Mutability::Mut, tcx.types.bool);
        let compute_cond = self.make_statement(
            StatementKind::Assign(
                cond.clone(),
                box Rvalue::BinaryOp(BinOp::Ne, Operand::Copy(end), Operand::Copy(beg))
            )
        );

        // `if end != beg { goto loop_body; } else { goto loop_end; }`
        self.block(
            vec![compute_cond],
            TerminatorKind::if_(tcx, Operand::Move(cond), loop_body, loop_end),
            is_cleanup
        );
    }

    fn make_usize(&self, value: u64) -> Box<Constant<'tcx>> {
        box Constant {
            span: self.span,
            ty: self.tcx.types.usize,
            user_ty: None,
            literal: ty::Const::from_usize(self.tcx, value),
        }
    }

    fn array_shim(&mut self, dest: Place<'tcx>, src: Place<'tcx>, ty: Ty<'tcx>, len: u64) {
        let tcx = self.tcx;
        let span = self.span;

        let beg = self.local_decls.push(temp_decl(Mutability::Mut, tcx.types.usize, span));
        let end = self.make_place(Mutability::Not, tcx.types.usize);

        // BB #0
        // `let mut beg = 0;`
        // `let end = len;`
        // `goto #1;`
        let inits = vec![
            self.make_statement(
                StatementKind::Assign(
                    Place::from(beg),
                    box Rvalue::Use(Operand::Constant(self.make_usize(0)))
                )
            ),
            self.make_statement(
                StatementKind::Assign(
                    end.clone(),
                    box Rvalue::Use(Operand::Constant(self.make_usize(len)))
                )
            )
        ];
        self.block(inits, TerminatorKind::Goto { target: BasicBlock::new(1) }, false);

        // BB #1: loop {
        //     BB #2;
        //     BB #3;
        // }
        // BB #4;
        self.loop_header(Place::from(beg),
                         end,
                         BasicBlock::new(2),
                         BasicBlock::new(4),
                         false);

        // BB #2
        // `dest[i] = Clone::clone(src[beg])`;
        // Goto #3 if ok, #5 if unwinding happens.
        let dest_field = dest.clone().index(beg);
        let src_field = src.index(beg);
        self.make_clone_call(dest_field, src_field, ty, BasicBlock::new(3),
                             BasicBlock::new(5));

        // BB #3
        // `beg = beg + 1;`
        // `goto #1`;
        let statements = vec![
            self.make_statement(
                StatementKind::Assign(
                    Place::from(beg),
                    box Rvalue::BinaryOp(
                        BinOp::Add,
                        Operand::Copy(Place::from(beg)),
                        Operand::Constant(self.make_usize(1))
                    )
                )
            )
        ];
        self.block(statements, TerminatorKind::Goto { target: BasicBlock::new(1) }, false);

        // BB #4
        // `return dest;`
        self.block(vec![], TerminatorKind::Return, false);

        // BB #5 (cleanup)
        // `let end = beg;`
        // `let mut beg = 0;`
        // goto #6;
        let end = beg;
        let beg = self.local_decls.push(temp_decl(Mutability::Mut, tcx.types.usize, span));
        let init = self.make_statement(
            StatementKind::Assign(
                Place::from(beg),
                box Rvalue::Use(Operand::Constant(self.make_usize(0)))
            )
        );
        self.block(vec![init], TerminatorKind::Goto { target: BasicBlock::new(6) }, true);

        // BB #6 (cleanup): loop {
        //     BB #7;
        //     BB #8;
        // }
        // BB #9;
        self.loop_header(Place::from(beg), Place::from(end),
                         BasicBlock::new(7), BasicBlock::new(9), true);

        // BB #7 (cleanup)
        // `drop(dest[beg])`;
        self.block(vec![], TerminatorKind::Drop {
            location: dest.index(beg),
            target: BasicBlock::new(8),
            unwind: None,
        }, true);

        // BB #8 (cleanup)
        // `beg = beg + 1;`
        // `goto #6;`
        let statement = self.make_statement(
            StatementKind::Assign(
                Place::from(beg),
                box Rvalue::BinaryOp(
                    BinOp::Add,
                    Operand::Copy(Place::from(beg)),
                    Operand::Constant(self.make_usize(1))
                )
            )
        );
        self.block(vec![statement], TerminatorKind::Goto { target: BasicBlock::new(6) }, true);

        // BB #9 (resume)
        self.block(vec![], TerminatorKind::Resume, true);
    }

    fn tuple_like_shim<I>(&mut self, dest: Place<'tcx>,
                          src: Place<'tcx>, tys: I)
            where I: Iterator<Item = Ty<'tcx>> {
        let mut previous_field = None;
        for (i, ity) in tys.enumerate() {
            let field = Field::new(i);
            let src_field = src.clone().field(field, ity);

            let dest_field = dest.clone().field(field, ity);

            // #(2i + 1) is the cleanup block for the previous clone operation
            let cleanup_block = self.block_index_offset(1);
            // #(2i + 2) is the next cloning block
            // (or the Return terminator if this is the last block)
            let next_block = self.block_index_offset(2);

            // BB #(2i)
            // `dest.i = Clone::clone(&src.i);`
            // Goto #(2i + 2) if ok, #(2i + 1) if unwinding happens.
            self.make_clone_call(
                dest_field.clone(),
                src_field,
                ity,
                next_block,
                cleanup_block,
            );

            // BB #(2i + 1) (cleanup)
            if let Some((previous_field, previous_cleanup)) = previous_field.take() {
                // Drop previous field and goto previous cleanup block.
                self.block(vec![], TerminatorKind::Drop {
                    location: previous_field,
                    target: previous_cleanup,
                    unwind: None,
                }, true);
            } else {
                // Nothing to drop, just resume.
                self.block(vec![], TerminatorKind::Resume, true);
            }

            previous_field = Some((dest_field, cleanup_block));
        }

        self.block(vec![], TerminatorKind::Return, false);
    }
}

/// Builds a "call" shim for `def_id`. The shim calls the
/// function specified by `call_kind`, first adjusting its first
/// argument according to `rcvr_adjustment`.
///
/// If `untuple_args` is a vec of types, the second argument of the
/// function will be untupled as these types.
fn build_call_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    rcvr_adjustment: Adjustment,
    call_kind: CallKind,
    untuple_args: Option<&[Ty<'tcx>]>,
) -> Body<'tcx> {
    debug!("build_call_shim(def_id={:?}, rcvr_adjustment={:?}, \
            call_kind={:?}, untuple_args={:?})",
           def_id, rcvr_adjustment, call_kind, untuple_args);

    let sig = tcx.fn_sig(def_id);
    let sig = tcx.erase_late_bound_regions(&sig);
    let span = tcx.def_span(def_id);

    debug!("build_call_shim: sig={:?}", sig);

    let mut local_decls = local_decls_for_sig(&sig, span);
    let source_info = SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE };

    let rcvr_arg = Local::new(1+0);
    let rcvr_l = Place::from(rcvr_arg);
    let mut statements = vec![];

    let rcvr = match rcvr_adjustment {
        Adjustment::Identity => Operand::Move(rcvr_l),
        Adjustment::Deref => Operand::Copy(rcvr_l.deref()),
        Adjustment::DerefMove => {
            // fn(Self, ...) -> fn(*mut Self, ...)
            let arg_ty = local_decls[rcvr_arg].ty;
            assert!(arg_ty.is_self());
            local_decls[rcvr_arg].ty = tcx.mk_mut_ptr(arg_ty);

            Operand::Move(rcvr_l.deref())
        }
        Adjustment::RefMut => {
            // let rcvr = &mut rcvr;
            let ref_rcvr = local_decls.push(temp_decl(
                Mutability::Not,
                tcx.mk_ref(tcx.lifetimes.re_erased, ty::TypeAndMut {
                    ty: sig.inputs()[0],
                    mutbl: hir::Mutability::MutMutable
                }),
                span
            ));
            let borrow_kind = BorrowKind::Mut {
                allow_two_phase_borrow: false,
            };
            statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(
                    Place::from(ref_rcvr),
                    box Rvalue::Ref(tcx.lifetimes.re_erased, borrow_kind, rcvr_l)
                )
            });
            Operand::Move(Place::from(ref_rcvr))
        }
    };

    let (callee, mut args) = match call_kind {
        CallKind::Indirect => (rcvr, vec![]),
        CallKind::Direct(def_id) => {
            let ty = tcx.type_of(def_id);
            (Operand::Constant(box Constant {
                span,
                ty,
                user_ty: None,
                literal: ty::Const::zero_sized(tcx, ty),
             }),
             vec![rcvr])
        }
    };

    if let Some(untuple_args) = untuple_args {
        args.extend(untuple_args.iter().enumerate().map(|(i, ity)| {
            let arg_place = Place::from(Local::new(1+1));
            Operand::Move(arg_place.field(Field::new(i), *ity))
        }));
    } else {
        args.extend((1..sig.inputs().len()).map(|i| {
            Operand::Move(Place::from(Local::new(1+i)))
        }));
    }

    let n_blocks = if let Adjustment::RefMut = rcvr_adjustment { 5 } else { 2 };
    let mut blocks = IndexVec::with_capacity(n_blocks);
    let block = |blocks: &mut IndexVec<_, _>, statements, kind, is_cleanup| {
        blocks.push(BasicBlockData {
            statements,
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup
        })
    };

    // BB #0
    block(&mut blocks, statements, TerminatorKind::Call {
        func: callee,
        args,
        destination: Some((Place::RETURN_PLACE,
                           BasicBlock::new(1))),
        cleanup: if let Adjustment::RefMut = rcvr_adjustment {
            Some(BasicBlock::new(3))
        } else {
            None
        },
        from_hir_call: true,
    }, false);

    if let Adjustment::RefMut = rcvr_adjustment {
        // BB #1 - drop for Self
        block(&mut blocks, vec![], TerminatorKind::Drop {
            location: Place::from(rcvr_arg),
            target: BasicBlock::new(2),
            unwind: None
        }, false);
    }
    // BB #1/#2 - return
    block(&mut blocks, vec![], TerminatorKind::Return, false);
    if let Adjustment::RefMut = rcvr_adjustment {
        // BB #3 - drop if closure panics
        block(&mut blocks, vec![], TerminatorKind::Drop {
            location: Place::from(rcvr_arg),
            target: BasicBlock::new(4),
            unwind: None
        }, true);

        // BB #4 - resume
        block(&mut blocks, vec![], TerminatorKind::Resume, true);
    }

    let mut body = Body::new(
        blocks,
        IndexVec::from_elem_n(
            SourceScopeData { span: span, parent_scope: None }, 1
        ),
        ClearCrossCrate::Clear,
        IndexVec::new(),
        None,
        local_decls,
        IndexVec::new(),
        sig.inputs().len(),
        vec![],
        span,
        vec![],
    );
    if let Abi::RustCall = sig.abi {
        body.spread_arg = Some(Local::new(sig.inputs().len()));
    }
    body
}

pub fn build_adt_ctor(tcx: TyCtxt<'_>, ctor_id: DefId) -> &Body<'_> {
    debug_assert!(tcx.is_constructor(ctor_id));

    let span = tcx.hir().span_if_local(ctor_id)
        .unwrap_or_else(|| bug!("no span for ctor {:?}", ctor_id));

    let param_env = tcx.param_env(ctor_id);

    // Normalize the sig.
    let sig = tcx.fn_sig(ctor_id)
        .no_bound_vars()
        .expect("LBR in ADT constructor signature");
    let sig = tcx.normalize_erasing_regions(param_env, sig);

    let (adt_def, substs) = match sig.output().sty {
        ty::Adt(adt_def, substs) => (adt_def, substs),
        _ => bug!("unexpected type for ADT ctor {:?}", sig.output())
    };

    debug!("build_ctor: ctor_id={:?} sig={:?}", ctor_id, sig);

    let local_decls = local_decls_for_sig(&sig, span);

    let source_info = SourceInfo {
        span,
        scope: OUTERMOST_SOURCE_SCOPE
    };

    let variant_index = if adt_def.is_enum() {
        adt_def.variant_index_with_ctor_id(ctor_id)
    } else {
        VariantIdx::new(0)
    };

    // Generate the following MIR:
    //
    // (return as Variant).field0 = arg0;
    // (return as Variant).field1 = arg1;
    //
    // return;
    debug!("build_ctor: variant_index={:?}", variant_index);

    let statements = expand_aggregate(
        Place::RETURN_PLACE,
        adt_def
            .variants[variant_index]
            .fields
            .iter()
            .enumerate()
            .map(|(idx, field_def)| (
                Operand::Move(Place::from(Local::new(idx + 1))),
                field_def.ty(tcx, substs),
            )),
        AggregateKind::Adt(adt_def, variant_index, substs, None, None),
        source_info,
    ).collect();

    let start_block = BasicBlockData {
        statements,
        terminator: Some(Terminator {
            source_info,
            kind: TerminatorKind::Return,
        }),
        is_cleanup: false
    };

    let body = Body::new(
        IndexVec::from_elem_n(start_block, 1),
        IndexVec::from_elem_n(
            SourceScopeData { span: span, parent_scope: None }, 1
        ),
        ClearCrossCrate::Clear,
        IndexVec::new(),
        None,
        local_decls,
        IndexVec::new(),
        sig.inputs().len(),
        vec![],
        span,
        vec![],
    );

    crate::util::dump_mir(
        tcx,
        None,
        "mir_map",
        &0,
        crate::transform::MirSource::item(ctor_id),
        &body,
        |_, _| Ok(()),
    );

    tcx.arena.alloc(body)
}
