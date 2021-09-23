use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_middle::mir::*;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::subst::{InternalSubsts, Subst};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_target::abi::VariantIdx;

use rustc_index::vec::{Idx, IndexVec};

use rustc_span::Span;
use rustc_target::spec::abi::Abi;

use std::fmt;
use std::iter;

use crate::util::expand_aggregate;
use crate::{
    abort_unwinding_calls, add_call_guards, add_moves_for_packed_drops, remove_noop_landing_pads,
    run_passes, simplify,
};
use rustc_middle::mir::patch::MirPatch;
use rustc_mir_dataflow::elaborate_drops::{self, DropElaborator, DropFlagMode, DropStyle};

pub fn provide(providers: &mut Providers) {
    providers.mir_shims = make_shim;
}

fn make_shim<'tcx>(tcx: TyCtxt<'tcx>, instance: ty::InstanceDef<'tcx>) -> Body<'tcx> {
    debug!("make_shim({:?})", instance);

    let mut result = match instance {
        ty::InstanceDef::Item(..) => bug!("item {:?} passed to make_shim", instance),
        ty::InstanceDef::VtableShim(def_id) => {
            build_call_shim(tcx, instance, Some(Adjustment::Deref), CallKind::Direct(def_id))
        }
        ty::InstanceDef::FnPtrShim(def_id, ty) => {
            let trait_ = tcx.trait_of_item(def_id).unwrap();
            let adjustment = match tcx.fn_trait_kind_from_lang_item(trait_) {
                Some(ty::ClosureKind::FnOnce) => Adjustment::Identity,
                Some(ty::ClosureKind::FnMut | ty::ClosureKind::Fn) => Adjustment::Deref,
                None => bug!("fn pointer {:?} is not an fn", ty),
            };

            build_call_shim(tcx, instance, Some(adjustment), CallKind::Indirect(ty))
        }
        // We are generating a call back to our def-id, which the
        // codegen backend knows to turn to an actual call, be it
        // a virtual call, or a direct call to a function for which
        // indirect calls must be codegen'd differently than direct ones
        // (such as `#[track_caller]`).
        ty::InstanceDef::ReifyShim(def_id) => {
            build_call_shim(tcx, instance, None, CallKind::Direct(def_id))
        }
        ty::InstanceDef::ClosureOnceShim { call_once: _, track_caller: _ } => {
            let fn_mut = tcx.require_lang_item(LangItem::FnMut, None);
            let call_mut = tcx
                .associated_items(fn_mut)
                .in_definition_order()
                .find(|it| it.kind == ty::AssocKind::Fn)
                .unwrap()
                .def_id;

            build_call_shim(tcx, instance, Some(Adjustment::RefMut), CallKind::Direct(call_mut))
        }
        ty::InstanceDef::DropGlue(def_id, ty) => build_drop_shim(tcx, def_id, ty),
        ty::InstanceDef::CloneShim(def_id, ty) => build_clone_shim(tcx, def_id, ty),
        ty::InstanceDef::Virtual(..) => {
            bug!("InstanceDef::Virtual ({:?}) is for direct calls only", instance)
        }
        ty::InstanceDef::Intrinsic(_) => {
            bug!("creating shims from intrinsics ({:?}) is unsupported", instance)
        }
    };
    debug!("make_shim({:?}) = untransformed {:?}", instance, result);

    run_passes(
        tcx,
        &mut result,
        MirPhase::Const,
        &[&[
            &add_moves_for_packed_drops::AddMovesForPackedDrops,
            &remove_noop_landing_pads::RemoveNoopLandingPads,
            &simplify::SimplifyCfg::new("make_shim"),
            &add_call_guards::CriticalCallEdges,
            &abort_unwinding_calls::AbortUnwindingCalls,
        ]],
    );

    debug!("make_shim({:?}) = {:?}", instance, result);

    result
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Adjustment {
    /// Pass the receiver as-is.
    Identity,

    /// We get passed `&[mut] self` and call the target with `*self`.
    ///
    /// This either copies `self` (if `Self: Copy`, eg. for function items), or moves out of it
    /// (for `VtableShim`, which effectively is passed `&own Self`).
    Deref,

    /// We get passed `self: Self` and call the target with `&mut self`.
    ///
    /// In this case we need to ensure that the `Self` is dropped after the call, as the callee
    /// won't do it for us.
    RefMut,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum CallKind<'tcx> {
    /// Call the `FnPtr` that was passed as the receiver.
    Indirect(Ty<'tcx>),

    /// Call a known `FnDef`.
    Direct(DefId),
}

fn local_decls_for_sig<'tcx>(
    sig: &ty::FnSig<'tcx>,
    span: Span,
) -> IndexVec<Local, LocalDecl<'tcx>> {
    iter::once(LocalDecl::new(sig.output(), span))
        .chain(sig.inputs().iter().map(|ity| LocalDecl::new(ity, span).immutable()))
        .collect()
}

fn build_drop_shim<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, ty: Option<Ty<'tcx>>) -> Body<'tcx> {
    debug!("build_drop_shim(def_id={:?}, ty={:?})", def_id, ty);

    // Check if this is a generator, if so, return the drop glue for it
    if let Some(&ty::Generator(gen_def_id, substs, _)) = ty.map(|ty| ty.kind()) {
        let body = tcx.optimized_mir(gen_def_id).generator_drop().unwrap();
        return body.clone().subst(tcx, substs);
    }

    let substs = if let Some(ty) = ty {
        tcx.intern_substs(&[ty.into()])
    } else {
        InternalSubsts::identity_for_item(tcx, def_id)
    };
    let sig = tcx.fn_sig(def_id).subst(tcx, substs);
    let sig = tcx.erase_late_bound_regions(sig);
    let span = tcx.def_span(def_id);

    let source_info = SourceInfo::outermost(span);

    let return_block = BasicBlock::new(1);
    let mut blocks = IndexVec::with_capacity(2);
    let block = |blocks: &mut IndexVec<_, _>, kind| {
        blocks.push(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup: false,
        })
    };
    block(&mut blocks, TerminatorKind::Goto { target: return_block });
    block(&mut blocks, TerminatorKind::Return);

    let source = MirSource::from_instance(ty::InstanceDef::DropGlue(def_id, ty));
    let mut body =
        new_body(tcx, source, blocks, local_decls_for_sig(&sig, span), sig.inputs().len(), span);

    if ty.is_some() {
        // The first argument (index 0), but add 1 for the return value.
        let dropee_ptr = Place::from(Local::new(1 + 0));
        if tcx.sess.opts.debugging_opts.mir_emit_retag {
            // Function arguments should be retagged, and we make this one raw.
            body.basic_blocks_mut()[START_BLOCK].statements.insert(
                0,
                Statement {
                    source_info,
                    kind: StatementKind::Retag(RetagKind::Raw, Box::new(dropee_ptr)),
                },
            );
        }
        let patch = {
            let param_env = tcx.param_env_reveal_all_normalized(def_id);
            let mut elaborator =
                DropShimElaborator { body: &body, patch: MirPatch::new(&body), tcx, param_env };
            let dropee = tcx.mk_place_deref(dropee_ptr);
            let resume_block = elaborator.patch.resume_block();
            elaborate_drops::elaborate_drop(
                &mut elaborator,
                source_info,
                dropee,
                (),
                return_block,
                elaborate_drops::Unwind::To(resume_block),
                START_BLOCK,
            );
            elaborator.patch
        };
        patch.apply(&mut body);
    }

    body
}

fn new_body<'tcx>(
    tcx: TyCtxt<'tcx>,
    source: MirSource<'tcx>,
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
    arg_count: usize,
    span: Span,
) -> Body<'tcx> {
    Body::new(
        tcx,
        source,
        basic_blocks,
        IndexVec::from_elem_n(
            SourceScopeData {
                span,
                parent_scope: None,
                inlined: None,
                inlined_parent_scope: None,
                local_data: ClearCrossCrate::Clear,
            },
            1,
        ),
        local_decls,
        IndexVec::new(),
        arg_count,
        vec![],
        span,
        None,
    )
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

    fn patch(&mut self) -> &mut MirPatch<'tcx> {
        &mut self.patch
    }
    fn body(&self) -> &'a Body<'tcx> {
        self.body
    }
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }

    fn drop_style(&self, _path: Self::Path, mode: DropFlagMode) -> DropStyle {
        match mode {
            DropFlagMode::Shallow => {
                // Drops for the contained fields are "shallow" and "static" - they will simply call
                // the field's own drop glue.
                DropStyle::Static
            }
            DropFlagMode::Deep => {
                // The top-level drop is "deep" and "open" - it will be elaborated to a drop ladder
                // dropping each field contained in the value.
                DropStyle::Open
            }
        }
    }

    fn get_drop_flag(&mut self, _path: Self::Path) -> Option<Operand<'tcx>> {
        None
    }

    fn clear_drop_flag(&mut self, _location: Location, _path: Self::Path, _mode: DropFlagMode) {}

    fn field_subpath(&self, _path: Self::Path, _field: Field) -> Option<Self::Path> {
        None
    }
    fn deref_subpath(&self, _path: Self::Path) -> Option<Self::Path> {
        None
    }
    fn downcast_subpath(&self, _path: Self::Path, _variant: VariantIdx) -> Option<Self::Path> {
        Some(())
    }
    fn array_subpath(&self, _path: Self::Path, _index: u64, _size: u64) -> Option<Self::Path> {
        None
    }
}

/// Builds a `Clone::clone` shim for `self_ty`. Here, `def_id` is `Clone::clone`.
fn build_clone_shim<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, self_ty: Ty<'tcx>) -> Body<'tcx> {
    debug!("build_clone_shim(def_id={:?})", def_id);

    let param_env = tcx.param_env(def_id);

    let mut builder = CloneShimBuilder::new(tcx, def_id, self_ty);
    let is_copy = self_ty.is_copy_modulo_regions(tcx.at(builder.span), param_env);

    let dest = Place::return_place();
    let src = tcx.mk_place_deref(Place::from(Local::new(1 + 0)));

    match self_ty.kind() {
        _ if is_copy => builder.copy_shim(),
        ty::Array(ty, len) => builder.array_shim(dest, src, ty, len),
        ty::Closure(_, substs) => {
            builder.tuple_like_shim(dest, src, substs.as_closure().upvar_tys())
        }
        ty::Tuple(..) => builder.tuple_like_shim(dest, src, self_ty.tuple_fields()),
        _ => bug!("clone shim for `{:?}` which is not `Copy` and is not an aggregate", self_ty),
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
        let sig = tcx.erase_late_bound_regions(sig);
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
        let source = MirSource::from_instance(ty::InstanceDef::CloneShim(
            self.def_id,
            self.sig.inputs_and_output[0],
        ));
        new_body(
            self.tcx,
            source,
            self.blocks,
            self.local_decls,
            self.sig.inputs().len(),
            self.span,
        )
    }

    fn source_info(&self) -> SourceInfo {
        SourceInfo::outermost(self.span)
    }

    fn block(
        &mut self,
        statements: Vec<Statement<'tcx>>,
        kind: TerminatorKind<'tcx>,
        is_cleanup: bool,
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
        Statement { source_info: self.source_info(), kind }
    }

    fn copy_shim(&mut self) {
        let rcvr = self.tcx.mk_place_deref(Place::from(Local::new(1 + 0)));
        let ret_statement = self.make_statement(StatementKind::Assign(Box::new((
            Place::return_place(),
            Rvalue::Use(Operand::Copy(rcvr)),
        ))));
        self.block(vec![ret_statement], TerminatorKind::Return, false);
    }

    fn make_place(&mut self, mutability: Mutability, ty: Ty<'tcx>) -> Place<'tcx> {
        let span = self.span;
        let mut local = LocalDecl::new(ty, span);
        if mutability == Mutability::Not {
            local = local.immutable();
        }
        Place::from(self.local_decls.push(local))
    }

    fn make_clone_call(
        &mut self,
        dest: Place<'tcx>,
        src: Place<'tcx>,
        ty: Ty<'tcx>,
        next: BasicBlock,
        cleanup: BasicBlock,
    ) {
        let tcx = self.tcx;

        let substs = tcx.mk_substs_trait(ty, &[]);

        // `func == Clone::clone(&ty) -> ty`
        let func_ty = tcx.mk_fn_def(self.def_id, substs);
        let func = Operand::Constant(Box::new(Constant {
            span: self.span,
            user_ty: None,
            literal: ty::Const::zero_sized(tcx, func_ty).into(),
        }));

        let ref_loc = self.make_place(
            Mutability::Not,
            tcx.mk_ref(tcx.lifetimes.re_erased, ty::TypeAndMut { ty, mutbl: hir::Mutability::Not }),
        );

        // `let ref_loc: &ty = &src;`
        let statement = self.make_statement(StatementKind::Assign(Box::new((
            ref_loc,
            Rvalue::Ref(tcx.lifetimes.re_erased, BorrowKind::Shared, src),
        ))));

        // `let loc = Clone::clone(ref_loc);`
        self.block(
            vec![statement],
            TerminatorKind::Call {
                func,
                args: vec![Operand::Move(ref_loc)],
                destination: Some((dest, next)),
                cleanup: Some(cleanup),
                from_hir_call: true,
                fn_span: self.span,
            },
            false,
        );
    }

    fn loop_header(
        &mut self,
        beg: Place<'tcx>,
        end: Place<'tcx>,
        loop_body: BasicBlock,
        loop_end: BasicBlock,
        is_cleanup: bool,
    ) {
        let tcx = self.tcx;

        let cond = self.make_place(Mutability::Mut, tcx.types.bool);
        let compute_cond = self.make_statement(StatementKind::Assign(Box::new((
            cond,
            Rvalue::BinaryOp(BinOp::Ne, Box::new((Operand::Copy(end), Operand::Copy(beg)))),
        ))));

        // `if end != beg { goto loop_body; } else { goto loop_end; }`
        self.block(
            vec![compute_cond],
            TerminatorKind::if_(tcx, Operand::Move(cond), loop_body, loop_end),
            is_cleanup,
        );
    }

    fn make_usize(&self, value: u64) -> Box<Constant<'tcx>> {
        Box::new(Constant {
            span: self.span,
            user_ty: None,
            literal: ty::Const::from_usize(self.tcx, value).into(),
        })
    }

    fn array_shim(
        &mut self,
        dest: Place<'tcx>,
        src: Place<'tcx>,
        ty: Ty<'tcx>,
        len: &'tcx ty::Const<'tcx>,
    ) {
        let tcx = self.tcx;
        let span = self.span;

        let beg = self.local_decls.push(LocalDecl::new(tcx.types.usize, span));
        let end = self.make_place(Mutability::Not, tcx.types.usize);

        // BB #0
        // `let mut beg = 0;`
        // `let end = len;`
        // `goto #1;`
        let inits = vec![
            self.make_statement(StatementKind::Assign(Box::new((
                Place::from(beg),
                Rvalue::Use(Operand::Constant(self.make_usize(0))),
            )))),
            self.make_statement(StatementKind::Assign(Box::new((
                end,
                Rvalue::Use(Operand::Constant(Box::new(Constant {
                    span: self.span,
                    user_ty: None,
                    literal: len.into(),
                }))),
            )))),
        ];
        self.block(inits, TerminatorKind::Goto { target: BasicBlock::new(1) }, false);

        // BB #1: loop {
        //     BB #2;
        //     BB #3;
        // }
        // BB #4;
        self.loop_header(Place::from(beg), end, BasicBlock::new(2), BasicBlock::new(4), false);

        // BB #2
        // `dest[i] = Clone::clone(src[beg])`;
        // Goto #3 if ok, #5 if unwinding happens.
        let dest_field = self.tcx.mk_place_index(dest, beg);
        let src_field = self.tcx.mk_place_index(src, beg);
        self.make_clone_call(dest_field, src_field, ty, BasicBlock::new(3), BasicBlock::new(5));

        // BB #3
        // `beg = beg + 1;`
        // `goto #1`;
        let statements = vec![self.make_statement(StatementKind::Assign(Box::new((
            Place::from(beg),
            Rvalue::BinaryOp(
                BinOp::Add,
                Box::new((Operand::Copy(Place::from(beg)), Operand::Constant(self.make_usize(1)))),
            ),
        ))))];
        self.block(statements, TerminatorKind::Goto { target: BasicBlock::new(1) }, false);

        // BB #4
        // `return dest;`
        self.block(vec![], TerminatorKind::Return, false);

        // BB #5 (cleanup)
        // `let end = beg;`
        // `let mut beg = 0;`
        // goto #6;
        let end = beg;
        let beg = self.local_decls.push(LocalDecl::new(tcx.types.usize, span));
        let init = self.make_statement(StatementKind::Assign(Box::new((
            Place::from(beg),
            Rvalue::Use(Operand::Constant(self.make_usize(0))),
        ))));
        self.block(vec![init], TerminatorKind::Goto { target: BasicBlock::new(6) }, true);

        // BB #6 (cleanup): loop {
        //     BB #7;
        //     BB #8;
        // }
        // BB #9;
        self.loop_header(
            Place::from(beg),
            Place::from(end),
            BasicBlock::new(7),
            BasicBlock::new(9),
            true,
        );

        // BB #7 (cleanup)
        // `drop(dest[beg])`;
        self.block(
            vec![],
            TerminatorKind::Drop {
                place: self.tcx.mk_place_index(dest, beg),
                target: BasicBlock::new(8),
                unwind: None,
            },
            true,
        );

        // BB #8 (cleanup)
        // `beg = beg + 1;`
        // `goto #6;`
        let statement = self.make_statement(StatementKind::Assign(Box::new((
            Place::from(beg),
            Rvalue::BinaryOp(
                BinOp::Add,
                Box::new((Operand::Copy(Place::from(beg)), Operand::Constant(self.make_usize(1)))),
            ),
        ))));
        self.block(vec![statement], TerminatorKind::Goto { target: BasicBlock::new(6) }, true);

        // BB #9 (resume)
        self.block(vec![], TerminatorKind::Resume, true);
    }

    fn tuple_like_shim<I>(&mut self, dest: Place<'tcx>, src: Place<'tcx>, tys: I)
    where
        I: Iterator<Item = Ty<'tcx>>,
    {
        let mut previous_field = None;
        for (i, ity) in tys.enumerate() {
            let field = Field::new(i);
            let src_field = self.tcx.mk_place_field(src, field, ity);

            let dest_field = self.tcx.mk_place_field(dest, field, ity);

            // #(2i + 1) is the cleanup block for the previous clone operation
            let cleanup_block = self.block_index_offset(1);
            // #(2i + 2) is the next cloning block
            // (or the Return terminator if this is the last block)
            let next_block = self.block_index_offset(2);

            // BB #(2i)
            // `dest.i = Clone::clone(&src.i);`
            // Goto #(2i + 2) if ok, #(2i + 1) if unwinding happens.
            self.make_clone_call(dest_field, src_field, ity, next_block, cleanup_block);

            // BB #(2i + 1) (cleanup)
            if let Some((previous_field, previous_cleanup)) = previous_field.take() {
                // Drop previous field and goto previous cleanup block.
                self.block(
                    vec![],
                    TerminatorKind::Drop {
                        place: previous_field,
                        target: previous_cleanup,
                        unwind: None,
                    },
                    true,
                );
            } else {
                // Nothing to drop, just resume.
                self.block(vec![], TerminatorKind::Resume, true);
            }

            previous_field = Some((dest_field, cleanup_block));
        }

        self.block(vec![], TerminatorKind::Return, false);
    }
}

/// Builds a "call" shim for `instance`. The shim calls the function specified by `call_kind`,
/// first adjusting its first argument according to `rcvr_adjustment`.
fn build_call_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::InstanceDef<'tcx>,
    rcvr_adjustment: Option<Adjustment>,
    call_kind: CallKind<'tcx>,
) -> Body<'tcx> {
    debug!(
        "build_call_shim(instance={:?}, rcvr_adjustment={:?}, call_kind={:?})",
        instance, rcvr_adjustment, call_kind
    );

    // `FnPtrShim` contains the fn pointer type that a call shim is being built for - this is used
    // to substitute into the signature of the shim. It is not necessary for users of this
    // MIR body to perform further substitutions (see `InstanceDef::has_polymorphic_mir_body`).
    let (sig_substs, untuple_args) = if let ty::InstanceDef::FnPtrShim(_, ty) = instance {
        let sig = tcx.erase_late_bound_regions(ty.fn_sig(tcx));

        let untuple_args = sig.inputs();

        // Create substitutions for the `Self` and `Args` generic parameters of the shim body.
        let arg_tup = tcx.mk_tup(untuple_args.iter());
        let sig_substs = tcx.mk_substs_trait(ty, &[ty::subst::GenericArg::from(arg_tup)]);

        (Some(sig_substs), Some(untuple_args))
    } else {
        (None, None)
    };

    let def_id = instance.def_id();
    let sig = tcx.fn_sig(def_id);
    let mut sig = tcx.erase_late_bound_regions(sig);

    assert_eq!(sig_substs.is_some(), !instance.has_polymorphic_mir_body());
    if let Some(sig_substs) = sig_substs {
        sig = sig.subst(tcx, sig_substs);
    }

    if let CallKind::Indirect(fnty) = call_kind {
        // `sig` determines our local decls, and thus the callee type in the `Call` terminator. This
        // can only be an `FnDef` or `FnPtr`, but currently will be `Self` since the types come from
        // the implemented `FnX` trait.

        // Apply the opposite adjustment to the MIR input.
        let mut inputs_and_output = sig.inputs_and_output.to_vec();

        // Initial signature is `fn(&? Self, Args) -> Self::Output` where `Args` is a tuple of the
        // fn arguments. `Self` may be passed via (im)mutable reference or by-value.
        assert_eq!(inputs_and_output.len(), 3);

        // `Self` is always the original fn type `ty`. The MIR call terminator is only defined for
        // `FnDef` and `FnPtr` callees, not the `Self` type param.
        let self_arg = &mut inputs_and_output[0];
        *self_arg = match rcvr_adjustment.unwrap() {
            Adjustment::Identity => fnty,
            Adjustment::Deref => tcx.mk_imm_ptr(fnty),
            Adjustment::RefMut => tcx.mk_mut_ptr(fnty),
        };
        sig.inputs_and_output = tcx.intern_type_list(&inputs_and_output);
    }

    // FIXME(eddyb) avoid having this snippet both here and in
    // `Instance::fn_sig` (introduce `InstanceDef::fn_sig`?).
    if let ty::InstanceDef::VtableShim(..) = instance {
        // Modify fn(self, ...) to fn(self: *mut Self, ...)
        let mut inputs_and_output = sig.inputs_and_output.to_vec();
        let self_arg = &mut inputs_and_output[0];
        debug_assert!(tcx.generics_of(def_id).has_self && *self_arg == tcx.types.self_param);
        *self_arg = tcx.mk_mut_ptr(*self_arg);
        sig.inputs_and_output = tcx.intern_type_list(&inputs_and_output);
    }

    let span = tcx.def_span(def_id);

    debug!("build_call_shim: sig={:?}", sig);

    let mut local_decls = local_decls_for_sig(&sig, span);
    let source_info = SourceInfo::outermost(span);

    let rcvr_place = || {
        assert!(rcvr_adjustment.is_some());
        Place::from(Local::new(1 + 0))
    };
    let mut statements = vec![];

    let rcvr = rcvr_adjustment.map(|rcvr_adjustment| match rcvr_adjustment {
        Adjustment::Identity => Operand::Move(rcvr_place()),
        Adjustment::Deref => Operand::Move(tcx.mk_place_deref(rcvr_place())),
        Adjustment::RefMut => {
            // let rcvr = &mut rcvr;
            let ref_rcvr = local_decls.push(
                LocalDecl::new(
                    tcx.mk_ref(
                        tcx.lifetimes.re_erased,
                        ty::TypeAndMut { ty: sig.inputs()[0], mutbl: hir::Mutability::Mut },
                    ),
                    span,
                )
                .immutable(),
            );
            let borrow_kind = BorrowKind::Mut { allow_two_phase_borrow: false };
            statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    Place::from(ref_rcvr),
                    Rvalue::Ref(tcx.lifetimes.re_erased, borrow_kind, rcvr_place()),
                ))),
            });
            Operand::Move(Place::from(ref_rcvr))
        }
    });

    let (callee, mut args) = match call_kind {
        // `FnPtr` call has no receiver. Args are untupled below.
        CallKind::Indirect(_) => (rcvr.unwrap(), vec![]),

        // `FnDef` call with optional receiver.
        CallKind::Direct(def_id) => {
            let ty = tcx.type_of(def_id);
            (
                Operand::Constant(Box::new(Constant {
                    span,
                    user_ty: None,
                    literal: ty::Const::zero_sized(tcx, ty).into(),
                })),
                rcvr.into_iter().collect::<Vec<_>>(),
            )
        }
    };

    let mut arg_range = 0..sig.inputs().len();

    // Take the `self` ("receiver") argument out of the range (it's adjusted above).
    if rcvr_adjustment.is_some() {
        arg_range.start += 1;
    }

    // Take the last argument, if we need to untuple it (handled below).
    if untuple_args.is_some() {
        arg_range.end -= 1;
    }

    // Pass all of the non-special arguments directly.
    args.extend(arg_range.map(|i| Operand::Move(Place::from(Local::new(1 + i)))));

    // Untuple the last argument, if we have to.
    if let Some(untuple_args) = untuple_args {
        let tuple_arg = Local::new(1 + (sig.inputs().len() - 1));
        args.extend(untuple_args.iter().enumerate().map(|(i, ity)| {
            Operand::Move(tcx.mk_place_field(Place::from(tuple_arg), Field::new(i), *ity))
        }));
    }

    let n_blocks = if let Some(Adjustment::RefMut) = rcvr_adjustment { 5 } else { 2 };
    let mut blocks = IndexVec::with_capacity(n_blocks);
    let block = |blocks: &mut IndexVec<_, _>, statements, kind, is_cleanup| {
        blocks.push(BasicBlockData {
            statements,
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup,
        })
    };

    // BB #0
    block(
        &mut blocks,
        statements,
        TerminatorKind::Call {
            func: callee,
            args,
            destination: Some((Place::return_place(), BasicBlock::new(1))),
            cleanup: if let Some(Adjustment::RefMut) = rcvr_adjustment {
                Some(BasicBlock::new(3))
            } else {
                None
            },
            from_hir_call: true,
            fn_span: span,
        },
        false,
    );

    if let Some(Adjustment::RefMut) = rcvr_adjustment {
        // BB #1 - drop for Self
        block(
            &mut blocks,
            vec![],
            TerminatorKind::Drop { place: rcvr_place(), target: BasicBlock::new(2), unwind: None },
            false,
        );
    }
    // BB #1/#2 - return
    block(&mut blocks, vec![], TerminatorKind::Return, false);
    if let Some(Adjustment::RefMut) = rcvr_adjustment {
        // BB #3 - drop if closure panics
        block(
            &mut blocks,
            vec![],
            TerminatorKind::Drop { place: rcvr_place(), target: BasicBlock::new(4), unwind: None },
            true,
        );

        // BB #4 - resume
        block(&mut blocks, vec![], TerminatorKind::Resume, true);
    }

    let mut body = new_body(
        tcx,
        MirSource::from_instance(instance),
        blocks,
        local_decls,
        sig.inputs().len(),
        span,
    );

    if let Abi::RustCall = sig.abi {
        body.spread_arg = Some(Local::new(sig.inputs().len()));
    }

    body
}

pub fn build_adt_ctor(tcx: TyCtxt<'_>, ctor_id: DefId) -> Body<'_> {
    debug_assert!(tcx.is_constructor(ctor_id));

    let span =
        tcx.hir().span_if_local(ctor_id).unwrap_or_else(|| bug!("no span for ctor {:?}", ctor_id));

    let param_env = tcx.param_env(ctor_id);

    // Normalize the sig.
    let sig = tcx.fn_sig(ctor_id).no_bound_vars().expect("LBR in ADT constructor signature");
    let sig = tcx.normalize_erasing_regions(param_env, sig);

    let (adt_def, substs) = match sig.output().kind() {
        ty::Adt(adt_def, substs) => (adt_def, substs),
        _ => bug!("unexpected type for ADT ctor {:?}", sig.output()),
    };

    debug!("build_ctor: ctor_id={:?} sig={:?}", ctor_id, sig);

    let local_decls = local_decls_for_sig(&sig, span);

    let source_info = SourceInfo::outermost(span);

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
        Place::return_place(),
        adt_def.variants[variant_index].fields.iter().enumerate().map(|(idx, field_def)| {
            (Operand::Move(Place::from(Local::new(idx + 1))), field_def.ty(tcx, substs))
        }),
        AggregateKind::Adt(adt_def, variant_index, substs, None, None),
        source_info,
        tcx,
    )
    .collect();

    let start_block = BasicBlockData {
        statements,
        terminator: Some(Terminator { source_info, kind: TerminatorKind::Return }),
        is_cleanup: false,
    };

    let source = MirSource::item(ctor_id);
    let body = new_body(
        tcx,
        source,
        IndexVec::from_elem_n(start_block, 1),
        local_decls,
        sig.inputs().len(),
        span,
    );

    rustc_middle::mir::dump_mir(tcx, None, "mir_map", &0, &body, |_, _| Ok(()));

    body
}
