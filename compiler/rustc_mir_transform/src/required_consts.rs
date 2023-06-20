//! Mono Item Collection
//! ====================
//!
//! This module is responsible for discovering all items that will contribute
//! to code generation of a single function. The important part here is that it not only
//! needs to find syntax-level items (functions, structs, etc) but also all
//! their monomorphized instantiations. Every non-generic, non-const function
//! maps to one LLVM artifact. Every generic function can produce
//! from zero to N artifacts, depending on the sets of type arguments it
//! is instantiated with.
//! This also applies to generic items from other crates: A generic definition
//! in crate X might produce monomorphizations that are compiled into crate Y.
//! We also have to collect these here.
//!
//! The following kinds of "mono items" are handled here:
//!
//! - Functions
//! - Methods
//! - Closures
//! - Statics
//! - Drop glue
//!
//! The following things also result in LLVM artifacts, but are not collected
//! here, since we instantiate them locally on demand when needed in a given
//! codegen unit:
//!
//! - Constants
//! - VTables
//! - Object Shims
//!
//! ### Finding neighbor nodes
//! Given an item's MIR, we can discover neighbors by walking the MIR and
//! any time we hit upon something that signifies a
//! reference to another mono item, we have found a neighbor.
//! The specific forms a reference to a neighboring node can take
//! in MIR are quite diverse. Here is an overview:
//!
//! #### Calling Functions/Methods
//! The most obvious form of one mono item referencing another is a
//! function or method call (represented by a CALL terminator in MIR). But
//! calls are not the only thing that might introduce a reference between two
//! function mono items, and as we will see below, they are just a
//! specialization of the form described next, and consequently will not get any
//! special treatment in the algorithm.
//!
//! #### Taking a reference to a function or method
//! A function does not need to actually be called in order to be a neighbor of
//! another function. It suffices to just take a reference in order to introduce
//! an edge. Consider the following example:
//!
//! ```
//! # use core::fmt::Display;
//! fn print_val<T: Display>(x: T) {
//!     println!("{}", x);
//! }
//!
//! fn call_fn(f: &dyn Fn(i32), x: i32) {
//!     f(x);
//! }
//!
//! fn main() {
//!     let print_i32 = print_val::<i32>;
//!     call_fn(&print_i32, 0);
//! }
//! ```
//! The MIR of none of these functions will contain an explicit call to
//! `print_val::<i32>`. Nonetheless, in order to mono this program, we need
//! an instance of this function. Thus, whenever we encounter a function or
//! method in operand position, we treat it as a neighbor of the current
//! mono item. Calls are just a special case of that.
//!
//! #### Drop glue
//! Drop glue mono items are introduced by MIR drop-statements. The
//! generated mono item will again have drop-glue item neighbors if the
//! type to be dropped contains nested values that also need to be dropped. It
//! might also have a function item neighbor for the explicit `Drop::drop`
//! implementation of its type.
//!
//! #### Unsizing Casts
//! A subtle way of introducing neighbor edges is by casting to a trait object.
//! Since the resulting fat-pointer contains a reference to a vtable, we need to
//! instantiate all object-safe methods of the trait, as we need to store
//! pointers to these functions even if they never get called anywhere. This can
//! be seen as a special case of taking a function reference.
//!
//! #### Boxes
//! Since `Box` expression have special compiler support, no explicit calls to
//! `exchange_malloc()` and `box_free()` may show up in MIR, even if the
//! compiler will generate them. We have to observe `Rvalue::Box` expressions
//! and Box-typed drop-statements for that purpose.
//!
//! Open Issues
//! -----------
//! Some things are not yet fully implemented in the current version of this
//! module.
//!
//! ### Const Fns
//! Ideally, no mono item should be generated for const fns unless there
//! is a call to them that cannot be evaluated at compile time. At the moment
//! this is not implemented however: a mono item will be produced
//! regardless of whether it is actually needed or not.

use rustc_hir::def::DefKind;
use rustc_hir::lang_items::LangItem;
use rustc_middle::mir::visit::TyContext;
use rustc_middle::mir::visit::Visitor as MirVisitor;
use rustc_middle::mir::MonoItem;
use rustc_middle::mir::MonoItems;
use rustc_middle::mir::{self, Local, Location};
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_span::def_id::DefId;
use rustc_span::source_map::Span;

pub struct MirNeighborCollector<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub body: &'a mir::Body<'tcx>,
    pub output: MonoItems<'tcx>,
    /// This is just the identity param env of the current MIR body cached here
    /// for convenience.
    pub param_env: ty::ParamEnv<'tcx>,
}

impl<'a, 'tcx> MirVisitor<'tcx> for MirNeighborCollector<'a, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        debug!("visiting rvalue {:?}", *rvalue);

        let span = self.body.source_info(location).span;

        match *rvalue {
            // When doing an cast from a regular pointer to a fat pointer, we
            // have to instantiate all methods of the trait being cast to, so we
            // can build the appropriate vtable.
            mir::Rvalue::Cast(
                mir::CastKind::Pointer(PointerCast::Unsize),
                ref operand,
                target_ty,
            )
            | mir::Rvalue::Cast(mir::CastKind::DynStar, ref operand, target_ty) => {
                self.output.push((
                    MonoItem::Vtable { source_ty: operand.ty(self.body, self.tcx), target_ty },
                    span,
                ));
            }
            mir::Rvalue::Cast(
                mir::CastKind::Pointer(PointerCast::ReifyFnPointer),
                ref operand,
                _,
            ) => {
                let fn_ty = operand.ty(self.body, self.tcx);
                self.visit_fn_use(fn_ty, false, span);
            }
            mir::Rvalue::Cast(
                mir::CastKind::Pointer(PointerCast::ClosureFnPointer(_)),
                ref operand,
                _,
            ) => {
                let source_ty = operand.ty(self.body, self.tcx);
                match *source_ty.kind() {
                    ty::Closure(def_id, substs) => {
                        if should_codegen_locally(self.tcx, def_id) {
                            self.output.push((MonoItem::Closure(def_id, substs), span));
                        }
                    }
                    _ => bug!(),
                }
            }
            mir::Rvalue::ThreadLocalRef(def_id) => {
                assert!(self.tcx.is_thread_local_static(def_id));
                if should_codegen_locally(self.tcx, def_id) {
                    trace!("collecting thread-local static {:?}", def_id);
                    self.output.push((MonoItem::Static(def_id), span));
                }
            }
            _ => { /* not interesting */ }
        }

        self.super_rvalue(rvalue, location);
    }

    /// This does not walk the constant, as it has been handled entirely here and trying
    /// to walk it would attempt to evaluate the `ty::Const` inside, which doesn't necessarily
    /// work, as some constants cannot be represented in the type system.
    #[instrument(skip(self), level = "debug")]
    fn visit_constant(&mut self, constant: &mir::Constant<'tcx>, location: Location) {
        self.output.push((MonoItem::Const(*constant), constant.span));
        MirVisitor::visit_ty(self, constant.literal.ty(), TyContext::Location(location));
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        debug!("visiting terminator {:?} @ {:?}", terminator, location);
        let source = self.body.source_info(location).span;

        let tcx = self.tcx;
        match terminator.kind {
            mir::TerminatorKind::Call { ref func, .. } => {
                let callee_ty = func.ty(self.body, tcx);
                self.visit_fn_use(callee_ty, true, source)
            }
            mir::TerminatorKind::Drop { ref place, .. } => {
                let ty = place.ty(self.body, self.tcx).ty;
                self.output.push((MonoItem::Drop(ty), source));
            }
            mir::TerminatorKind::InlineAsm { ref operands, .. } => {
                for op in operands {
                    match *op {
                        mir::InlineAsmOperand::SymFn { ref value } => {
                            self.visit_fn_use(value.literal.ty(), false, source);
                        }
                        mir::InlineAsmOperand::SymStatic { def_id } => {
                            if should_codegen_locally(self.tcx, def_id) {
                                trace!("collecting asm sym static {:?}", def_id);
                                self.output.push((MonoItem::Static(def_id), source));
                            }
                        }
                        _ => {}
                    }
                }
            }
            mir::TerminatorKind::Assert { ref msg, .. } => {
                let lang_item = match &**msg {
                    mir::AssertKind::BoundsCheck { .. } => LangItem::PanicBoundsCheck,
                    _ => LangItem::Panic,
                };
                let instance = Instance::mono(tcx, tcx.require_lang_item(lang_item, Some(source)));
                if should_codegen_locally(tcx, instance.def_id()) {
                    self.output
                        .push((MonoItem::Fn(instance.def_id(), instance.substs, true), source));
                }
            }
            mir::TerminatorKind::Terminate { .. } => {
                let instance = Instance::mono(
                    tcx,
                    tcx.require_lang_item(LangItem::PanicCannotUnwind, Some(source)),
                );
                if should_codegen_locally(tcx, instance.def_id()) {
                    self.output
                        .push((MonoItem::Fn(instance.def_id(), instance.substs, true), source));
                }
            }
            // The contents of these are walked with `super_terminator` below. They don't have any
            // special constants or call-like behavior.
            mir::TerminatorKind::Yield { .. }
            | mir::TerminatorKind::Goto { .. }
            | mir::TerminatorKind::SwitchInt { .. }
            | mir::TerminatorKind::Resume
            | mir::TerminatorKind::Return
            | mir::TerminatorKind::FalseUnwind { .. }
            | mir::TerminatorKind::FalseEdge { .. }
            | mir::TerminatorKind::Unreachable
            | mir::TerminatorKind::GeneratorDrop => {}
        }

        if let Some(mir::UnwindAction::Terminate) = terminator.unwind() {
            let instance = Instance::mono(
                tcx,
                tcx.require_lang_item(LangItem::PanicCannotUnwind, Some(source)),
            );
            if should_codegen_locally(tcx, instance.def_id()) {
                self.output.push((MonoItem::Fn(instance.def_id(), instance.substs, true), source));
            }
        }

        self.super_terminator(terminator, location);
    }

    fn visit_local(
        &mut self,
        _place_local: Local,
        _context: mir::visit::PlaceContext,
        _location: Location,
    ) {
    }
}

impl<'a, 'tcx> MirNeighborCollector<'a, 'tcx> {
    fn visit_fn_use(&mut self, ty: Ty<'tcx>, is_direct_call: bool, source: Span) {
        if let ty::FnDef(def_id, substs) = *ty.kind() {
            let should_codegen_locally = || {
                if self.tcx.try_normalize_erasing_regions(self.param_env, substs).is_err() {
                    return true;
                }
                if let Ok(Some(instance)) =
                    ty::Instance::resolve(self.tcx, self.param_env, def_id, substs)
                {
                    if let ty::InstanceDef::Item(..) = instance.def {
                        should_codegen_locally(self.tcx, instance.def_id())
                    } else {
                        true
                    }
                } else {
                    true
                }
            };
            if !is_direct_call || should_codegen_locally() {
                self.output.push((MonoItem::Fn(def_id, substs, is_direct_call), source))
            }
        }
    }
}

/// Returns `true` if we should codegen an item in the local crate, or returns `false` if we
/// can just link to the upstream crate and therefore don't need a mono item.
fn should_codegen_locally<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    if tcx.is_foreign_item(def_id) {
        // Foreign items are always linked against, there's no way of instantiating them.
        return false;
    }

    if def_id.is_local() {
        // Local items cannot be referred to locally without monomorphizing them locally.
        return true;
    }

    if tcx.is_reachable_non_generic(def_id) {
        // We can link to the item in question, no instance needed in this crate.
        return false;
    }

    if let DefKind::Static(_) = tcx.def_kind(def_id) {
        // We cannot monomorphize statics from upstream crates.
        return false;
    }

    tcx.is_mir_available(def_id)
}
