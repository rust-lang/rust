//! Speculative shadow execution for `#[rad_protected(shadow_only)]` functions.
//!
//! This is the function-level baseline of the Rad-Rust rollback work: a protected function is
//! turned into a transaction over a *shadow* copy of the state its caller lent it.
//!
//! ```ignore (illustrative)
//! fn update(state: &mut State) {
//!     state.position.x += 1;
//! }
//! ```
//!
//! becomes, conceptually:
//!
//! ```ignore (illustrative)
//! fn update(state: &mut State) {
//!     let original = state;
//!     let mut shadow = *original;      // eager copy of the whole state
//!     {
//!         let state = &mut shadow;     // the body now runs on the shadow
//!         state.position.x += 1;
//!     }
//!     original.position.x = shadow.position.x;   // commit only what may have been written
//! }
//! ```
//!
//! The body never touches the caller's `State`, so the original stays intact and available for a
//! rollback, and only the fields the body may write are copied back. `position.y` and `counter`
//! are left alone.
//!
//! The redirection is done by rebinding the argument local itself. The body already reaches the
//! state through `_1`, so after `_1 = &mut _shadow` the very same MIR `Place` `(*_1).position.x`
//! denotes `_shadow.position.x`, and no use inside the body has to be rewritten.
//!
//! What this pass deliberately does *not* do yet:
//!
//!   * decide whether to commit. The commit here is unconditional; a later commit adds the
//!     `Commit`/`Abort` decision, and the comparison between redundant executions belongs to the
//!     voting layer, not here. There is no fake voter.
//!   * compare the original against the shadow. With a single execution that would only report
//!     whether the function wrote anything, and a bytewise comparison of an arbitrary Rust
//!     aggregate is not sound anyway (padding, uninit bytes).
//!   * track *runtime* dirtiness. The set computed here is a conservative compile-time may-dirty
//!     set: a field is in it if some path may write it. Committing an unchanged field back is
//!     harmless. Path-sensitive dirty bits and first-write (copy-on-write) copying come later.
//!
//! The eligibility rules are correspondingly narrow; see [`ShadowedArg::find`]. Anything outside
//! them leaves the function untouched, with the reason reported on stderr.

use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_hir::{Mutability, find_attr};
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    BasicBlockData, Body, BorrowKind, Local, LocalDecl, Location, MutBorrowKind, Operand, Place,
    PlaceElem, ProjectionElem, RETURN_PLACE, Rvalue, START_BLOCK, SourceInfo, Statement,
    StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, List, Ty, TyCtxt, TypingEnv};

pub(super) struct RadProtectedFunctionShadow;

impl<'tcx> crate::MirPass<'tcx> for RadProtectedFunctionShadow {
    fn is_enabled(&self, _sess: &rustc_session::Session) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();

        if !find_attr!(tcx, def_id, RadProtectedShadow(_)) {
            return;
        }

        with_no_trimmed_paths!({
            eprintln!("\n=== Shadow Execution for {} ===", tcx.def_path_str(def_id));

            let arg = match ShadowedArg::find(tcx, body) {
                Ok(arg) => arg,
                Err(reason) => {
                    eprintln!("not shadowed: {reason}");
                    eprintln!("================================");
                    return;
                }
            };

            let dirty = match BodyScan::run(tcx, body, arg.local) {
                Ok(dirty) => dirty,
                Err(reason) => {
                    eprintln!("not shadowed: {reason}");
                    eprintln!("================================");
                    return;
                }
            };

            if dirty.is_empty() {
                eprintln!("not shadowed: the body never writes through {:?}", arg.local);
                eprintln!("================================");
                return;
            }

            let locals = apply(tcx, body, &arg, &dirty);

            eprintln!("shadowed argument: {:?}: &mut {}", arg.local, arg.pointee);
            eprintln!("original: {:?}, shadow: {:?}", locals.original, locals.shadow);
            eprintln!("may-dirty commit set:");
            for proj in &dirty {
                eprintln!(
                    "\t{:?} = copy {:?}",
                    original_place(tcx, locals.original, proj),
                    Place { local: locals.shadow, projection: proj }
                );
            }
            eprintln!("================================");
        });
    }

    fn is_required(&self) -> bool {
        true
    }
}

/// The single `&mut T` argument whose pointee gets shadowed.
struct ShadowedArg<'tcx> {
    local: Local,
    /// `&mut T`, reused verbatim for the saved-original and shadow-facing locals.
    ref_ty: Ty<'tcx>,
    /// `T`, the type of the shadow copy.
    pointee: Ty<'tcx>,
}

impl<'tcx> ShadowedArg<'tcx> {
    /// Picks the argument to shadow, or explains why this function is out of scope.
    ///
    /// The first version handles exactly one `fn protected(state: &mut T) -> R`:
    ///
    ///   * exactly one `&mut` argument, and no other argument carrying a `&mut`, a raw pointer, or
    ///     a `&` to something with interior mutability, so there is a single root to rebase writes
    ///     onto and no second way to reach the caller's memory;
    ///   * `T: Sized`, because the shadow is a local of type `T`;
    ///   * `T: Copy`, so the eager copy in and the field-by-field copy back need no drop of the
    ///     overwritten value and no deep copy of owned data. Compiler-injected duplication of an
    ///     owned value is not automatically sound just because the input was safe Rust, so
    ///     non-`Copy` state is rejected rather than byte-copied;
    ///   * no reference or raw pointer *inside* `T` at all. A `&mut` or raw pointer could be used
    ///     to write memory the shadow does not cover; a `&` could be aimed at an `UnsafeCell` and
    ///     written through just as easily; and any reference-typed field can be overwritten by the
    ///     body with a borrow of the shadow, which the commit would then publish into the caller's
    ///     state, leaving it pointing at a local that dies at the return;
    ///   * no union anywhere in `T`, since union fields overlap and the commit reads each
    ///     may-dirty field back at that field's own type;
    ///   * a return type free of references, since any reference into the shadow would dangle once
    ///     the shadow goes away.
    fn find(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Result<Self, String> {
        let typing_env = body.typing_env(tcx);
        let mut found: Option<Self> = None;

        for local in body.args_iter() {
            let ty = body.local_decls[local].ty;

            if let ty::Ref(_, pointee, Mutability::Mut) = ty.kind() {
                if let Some(previous) = &found {
                    return Err(format!(
                        "more than one `&mut` argument ({:?} and {local:?}); rebasing writes onto \
                         several roots comes later",
                        previous.local
                    ));
                }
                found = Some(Self { local, ref_ty: ty, pointee: *pointee });
                continue;
            }

            if let Some(blocker) = ReachesPointer::writable(tcx, typing_env, ty) {
                return Err(match blocker {
                    Blocker::Pointer(inner) => format!(
                        "argument {local:?}: {ty} reaches a `&mut` or raw pointer ({inner}) that \
                         the shadow does not cover"
                    ),
                    Blocker::Interior(inner) => format!(
                        "argument {local:?}: {ty} reaches {inner}, whose referent has interior \
                         mutability, so the body could write through it without a `&mut` and the \
                         write would escape the shadow"
                    ),
                    Blocker::Union(inner) => format!(
                        "argument {local:?}: {ty} contains the union {inner}, whose fields overlap"
                    ),
                    Blocker::Opaque(inner) => format!(
                        "argument {local:?}: {ty} contains {inner}, which cannot be inspected here"
                    ),
                });
            }
        }

        let arg = found.ok_or_else(|| "no `&mut` argument to shadow".to_string())?;

        if !arg.pointee.is_sized(tcx, typing_env) {
            return Err(format!("{} is unsized, so it cannot be a shadow local", arg.pointee));
        }

        if !tcx.type_is_copy_modulo_regions(typing_env, arg.pointee) {
            return Err(format!(
                "{} is not `Copy`; duplicating and committing owned state needs the deep-copy and \
                 drop handling that a later commit adds",
                arg.pointee
            ));
        }

        // The state is checked with the strict rule. A `&T` inside it is not only a way to reach
        // memory the shadow does not own: the body can also overwrite that field with a borrow of
        // the shadow, which the commit would then publish into the caller's state, leaving it
        // holding a reference into a local that dies at the return.
        if let Some(blocker) = ReachesPointer::any(tcx, typing_env, arg.pointee) {
            let pointee = arg.pointee;
            return Err(match blocker {
                Blocker::Pointer(inner) => format!(
                    "{pointee} reaches a reference or raw pointer ({inner}); writes through it \
                     would escape the shadow, and a reference field could be overwritten with a \
                     borrow of the shadow and published by the commit"
                ),
                Blocker::Interior(inner) => format!(
                    "{pointee} reaches {inner}, whose referent has interior mutability, so writes \
                     through it would escape the shadow"
                ),
                Blocker::Union(inner) => format!(
                    "{pointee} contains the union {inner}: its fields overlap, so committing one \
                     may-dirty field would read it at a type the union does not currently hold"
                ),
                Blocker::Opaque(inner) => format!(
                    "{pointee} contains {inner}, which cannot be inspected here, so the shadow \
                     cannot be shown to cover the whole state"
                ),
            });
        }

        let ret_ty = body.local_decls[RETURN_PLACE].ty;
        if let Some(blocker) = ReachesPointer::any(tcx, typing_env, ret_ty) {
            return Err(match blocker {
                Blocker::Pointer(inner) => format!(
                    "return type {ret_ty} contains a reference ({inner}), which could point into \
                     the shadow and dangle after it is dropped"
                ),
                Blocker::Interior(inner) => format!(
                    "return type {ret_ty} contains {inner}, whose referent has interior \
                     mutability, so it could alias the shadow"
                ),
                Blocker::Union(inner) => format!(
                    "return type {ret_ty} contains the union {inner}, whose fields overlap"
                ),
                Blocker::Opaque(inner) => format!(
                    "return type {ret_ty} contains {inner}, which cannot be inspected here, so it \
                     cannot be shown not to borrow from the shadow"
                ),
            });
        }

        Ok(arg)
    }
}

/// Why a type cannot be shown to be self-contained, and the exact inner type responsible.
enum Blocker<'tcx> {
    /// A pointer through which memory outside the shadow is reachable.
    Pointer(Ty<'tcx>),
    /// A shared reference to a type with interior mutability. `&T` is normally harmless here, but
    /// an `UnsafeCell` behind one can be written through without a `&mut`, so a write would land
    /// on the caller's memory instead of the shadow.
    Interior(Ty<'tcx>),
    /// A union, whose fields overlap. The commit reads back each may-dirty field at that field's
    /// own type, which is only valid if the union currently holds that variant; the pass cannot
    /// tell which one it holds.
    Union(Ty<'tcx>),
    /// A type this pass cannot look inside - a generic parameter before monomorphization, a trait
    /// object, a closure - so it has to assume the worst.
    Opaque(Ty<'tcx>),
}

/// How strict to be about shared references in a given position.
#[derive(Clone, Copy, PartialEq)]
enum Refs {
    /// Any reference at all blocks. Used for the shadowed state and for the return type: a `&` in
    /// the state can be overwritten with a borrow of the shadow and then published by the commit,
    /// and a `&` in the return type would point into the shadow and dangle once it is gone.
    None,
    /// `&mut` and raw pointers block; a `&T` blocks only when `T` reaches an `UnsafeCell`. Used
    /// for the arguments that are not being shadowed, where an ordinary `&str` or `&u32` is a
    /// read-only input that cannot escape the shadow or be written through.
    SharedUnlessInterior,
}

/// Looks for a way for `ty` to reach memory that the shadow copy does not own.
///
/// `Ty::walk` only visits generic arguments, not the fields of an ADT, so this recurses through
/// field types itself. Anything it cannot decompose is a [`Blocker::Opaque`], so an unfamiliar
/// type leaves the function out of scope rather than being silently shadowed.
struct ReachesPointer<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: TypingEnv<'tcx>,
    /// How shared references are treated in this position; see [`Refs`].
    refs: Refs,
    /// A `Copy` type can still be recursive through references, so cycles have to be cut.
    seen: FxHashSet<Ty<'tcx>>,
}

impl<'tcx> ReachesPointer<'tcx> {
    /// For an argument that is not the one being shadowed: it is a way *into* memory the shadow
    /// does not cover only if it can be written through.
    fn writable(
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<Blocker<'tcx>> {
        Self { tcx, typing_env, refs: Refs::SharedUnlessInterior, seen: FxHashSet::default() }
            .check(ty)
    }

    /// Any reference at all, shared ones included. For the shadowed state and the return type.
    fn any(tcx: TyCtxt<'tcx>, typing_env: TypingEnv<'tcx>, ty: Ty<'tcx>) -> Option<Blocker<'tcx>> {
        Self { tcx, typing_env, refs: Refs::None, seen: FxHashSet::default() }.check(ty)
    }

    /// Whether `ty` reaches an `UnsafeCell`, i.e. can be mutated through a `&`.
    fn has_interior_mutability(&mut self, ty: Ty<'tcx>) -> bool {
        !ty.is_freeze(self.tcx, self.typing_env)
    }

    fn check(&mut self, ty: Ty<'tcx>) -> Option<Blocker<'tcx>> {
        let ty = self.tcx.normalize_erasing_regions(self.typing_env, ty);

        if !self.seen.insert(ty) {
            return None;
        }

        match *ty.kind() {
            // A raw pointer counts whatever its mutability, since it can be cast to `*mut`.
            ty::RawPtr(..) | ty::Ref(_, _, Mutability::Mut) => Some(Blocker::Pointer(ty)),
            ty::Ref(..) if self.refs == Refs::None => Some(Blocker::Pointer(ty)),
            // `&T` where `T` has interior mutability: `Cell::set` and friends write through it
            // without ever taking a `&mut`, so the write escapes the shadow.
            ty::Ref(_, pointee, Mutability::Not) if self.has_interior_mutability(pointee) => {
                Some(Blocker::Interior(ty))
            }
            ty::Ref(..) => None,

            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Never
            // A function pointer is a value, not a way into memory. Calling through one with the
            // shadow reference would need a reborrow, which the body scan rejects.
            | ty::FnPtr(..)
            | ty::FnDef(..) => None,

            ty::Array(elem, _) | ty::Slice(elem) | ty::Pat(elem, _) => self.check(elem),
            ty::Tuple(fields) => fields.iter().find_map(|field| self.check(field)),
            // Union fields overlap, so a may-dirty projection naming one of them is not an
            // independent piece of state: committing `.flag` reads the union at `bool` even when
            // the body left a `u8` in it. Widening to the whole union would fix that, but the
            // pass would still have to prove the body never leaves an inactive field dirty, so
            // unions stay out of scope for now.
            ty::Adt(def, _) if def.is_union() => Some(Blocker::Union(ty)),
            ty::Adt(def, args) => {
                def.all_fields().find_map(|field| self.check(field.ty(self.tcx, args)))
            }

            _ => Some(Blocker::Opaque(ty)),
        }
    }
}

/// Walks the original body to collect the compile-time may-dirty set and to reject bodies whose
/// writes the set would not describe faithfully.
struct BodyScan<'tcx> {
    tcx: TyCtxt<'tcx>,
    arg: Local,
    /// Projections *relative to the pointee*: `(*_1).position.x` is recorded as `.position.x`, so
    /// the same projection can be rebased onto both the shadow local and the saved original.
    dirty: FxIndexSet<&'tcx List<PlaceElem<'tcx>>>,
    rejected: Option<String>,
}

impl<'tcx> BodyScan<'tcx> {
    fn run(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        arg: Local,
    ) -> Result<Vec<&'tcx List<PlaceElem<'tcx>>>, String> {
        let mut scan = Self { tcx, arg, dirty: FxIndexSet::default(), rejected: None };
        scan.visit_body(body);

        if let Some(reason) = scan.rejected {
            return Err(reason);
        }

        Ok(narrow_to_outermost(scan.dirty))
    }

    fn reject(&mut self, reason: String) {
        self.rejected.get_or_insert(reason);
    }

    fn record_write(&mut self, place: Place<'tcx>) {
        if place.local != self.arg {
            return;
        }

        // `visit_place` already rejects any use of the argument that is not through `*_1`.
        if let Some(projection) = self.relative_projection(place) {
            self.dirty.insert(projection);
        }
    }

    /// The part of `place`'s projection that applies to the pointee, cut off at the first element
    /// the commit cannot spell as a fixed field path.
    ///
    /// Truncating *widens* the committed region — `(*_1).samples[i] = v` commits the whole
    /// `samples` array — which keeps the set conservative rather than unsound.
    fn relative_projection(&self, place: Place<'tcx>) -> Option<&'tcx List<PlaceElem<'tcx>>> {
        let [ProjectionElem::Deref, rest @ ..] = &place.projection[..] else {
            return None;
        };

        let fields =
            rest.iter().take_while(|elem| matches!(elem, ProjectionElem::Field(..))).count();
        Some(self.tcx.mk_place_elems(&rest[..fields]))
    }
}

impl<'tcx> Visitor<'tcx> for BodyScan<'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        if place.local == self.arg {
            if matches!(place.projection.first(), Some(ProjectionElem::Deref)) {
                if is_write(context) {
                    self.record_write(*place);
                }
            } else if context.is_use() {
                // Rebinding `_1` redirects the body only where it reads *through* `_1`. A copy or
                // move of the reference itself would create an alias whose writes this scan does
                // not see.
                self.reject(format!(
                    "{:?} is used as a whole reference at {location:?}, not through `*{:?}`",
                    self.arg, self.arg
                ));
            }
        }

        self.super_place(place, context, location);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        match rvalue {
            // A mutable reborrow or raw pointer hands out a write capability that the may-dirty
            // set cannot follow, so the commit could miss a field. Shared borrows are fine: they
            // cannot write, and borrowck already stopped one from outliving the body.
            Rvalue::Ref(_, BorrowKind::Mut { .. }, place) if place.local == self.arg => {
                self.reject(format!(
                    "{place:?} is mutably reborrowed at {location:?}; writes through the reborrow \
                     are not tracked"
                ));
            }
            Rvalue::RawPtr(_, place) if place.local == self.arg => {
                self.reject(format!(
                    "{place:?} has its address taken at {location:?}; writes through the pointer \
                     are not tracked"
                ));
            }
            _ => {}
        }

        self.super_rvalue(rvalue, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let TerminatorKind::TailCall { .. } = terminator.kind {
            self.reject(format!("tail call at {location:?} leaves no `Return` to commit at"));
        }

        self.super_terminator(terminator, location);
    }
}

fn is_write(context: PlaceContext) -> bool {
    matches!(
        context,
        PlaceContext::MutatingUse(
            MutatingUseContext::Store
                | MutatingUseContext::Call
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Yield
                | MutatingUseContext::SetDiscriminant
                | MutatingUseContext::Drop
        )
    )
}

/// Drops every projection that a shorter one already covers, so that
/// `[.position, .position.x]` commits `.position` once instead of writing the parent and then a
/// field of it.
fn narrow_to_outermost<'tcx>(
    dirty: FxIndexSet<&'tcx List<PlaceElem<'tcx>>>,
) -> Vec<&'tcx List<PlaceElem<'tcx>>> {
    // `FxIndexSet` is insertion-ordered, so the result follows the deterministic walk order of the
    // body rather than a hash order.
    dirty
        .iter()
        .copied()
        .filter(|projection| {
            !dirty.iter().any(|other| *other != *projection && is_prefix_of(other, projection))
        })
        .collect()
}

fn is_prefix_of<'tcx>(prefix: &List<PlaceElem<'tcx>>, of: &List<PlaceElem<'tcx>>) -> bool {
    prefix.len() <= of.len() && prefix[..] == of[..prefix.len()]
}

struct ShadowLocals {
    /// Holds the caller's reference for the commit, and for the rollback a later commit adds.
    original: Local,
    /// The speculative copy the body runs on.
    shadow: Local,
}

/// `(*_original).<projection>`, the commit destination for one may-dirty projection.
fn original_place<'tcx>(
    tcx: TyCtxt<'tcx>,
    original: Local,
    projection: &'tcx List<PlaceElem<'tcx>>,
) -> Place<'tcx> {
    Place::from(original)
        .project_deeper(&[ProjectionElem::Deref], tcx)
        .project_deeper(projection, tcx)
}

fn apply<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    arg: &ShadowedArg<'tcx>,
    dirty: &[&'tcx List<PlaceElem<'tcx>>],
) -> ShadowLocals {
    let span = body.span;
    let source_info = SourceInfo::outermost(span);

    let assign = |place, rvalue| {
        Statement::new(source_info, StatementKind::Assign(Box::new((place, rvalue))))
    };

    let original = body.local_decls.push(LocalDecl::new(arg.ref_ty, span));
    let shadow = body.local_decls.push(LocalDecl::new(arg.pointee, span));
    let shadow_ref = body.local_decls.push(LocalDecl::new(arg.ref_ty, span));

    // The argument local is rebound below, so it has to be a mutable slot.
    body.local_decls[arg.local].mutability = Mutability::Mut;

    let setup = vec![
        // _original = move _1;
        assign(Place::from(original), Rvalue::Use(Operand::Move(Place::from(arg.local)))),
        // _shadow = copy (*_original);
        assign(
            Place::from(shadow),
            Rvalue::Use(Operand::Copy(
                Place::from(original).project_deeper(&[ProjectionElem::Deref], tcx),
            )),
        ),
        // _shadow_ref = &mut _shadow;
        assign(
            Place::from(shadow_ref),
            Rvalue::Ref(
                tcx.lifetimes.re_erased,
                BorrowKind::Mut { kind: MutBorrowKind::Default },
                Place::from(shadow),
            ),
        ),
        // _1 = move _shadow_ref; from here the body's own `(*_1)...` places reach the shadow.
        assign(Place::from(arg.local), Rvalue::Use(Operand::Move(Place::from(shadow_ref)))),
    ];

    // The start block must stay predecessor-free, so the original entry block is moved aside and
    // the setup takes its place.
    let entry = body.basic_blocks_mut()[START_BLOCK].clone();
    let body_entry = body.basic_blocks_mut().push(entry);
    body.basic_blocks_mut()[START_BLOCK] = BasicBlockData::new_stmts(
        setup,
        Some(Terminator { source_info, kind: TerminatorKind::Goto { target: body_entry } }),
        false,
    );

    // TODO(voting): the comparison/voting layer will decide whether the shadow is committed or
    // discarded. Until it exists this baseline always commits, and an unwind out of the body
    // still commits nothing, leaving the caller's state as it was.
    let commit: Vec<Statement<'tcx>> = dirty
        .iter()
        .map(|projection| {
            assign(
                original_place(tcx, original, projection),
                Rvalue::Use(Operand::Copy(Place { local: shadow, projection })),
            )
        })
        .collect();

    for block in body.basic_blocks_mut() {
        if matches!(block.terminator().kind, TerminatorKind::Return) {
            block.statements.extend(commit.iter().cloned());
        }
    }

    ShadowLocals { original, shadow }
}
