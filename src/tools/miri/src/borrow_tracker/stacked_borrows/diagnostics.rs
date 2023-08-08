use smallvec::SmallVec;
use std::fmt;

use rustc_middle::mir::interpret::{alloc_range, AllocId, AllocRange, InterpError};
use rustc_span::{Span, SpanData};
use rustc_target::abi::Size;

use crate::borrow_tracker::{
    stacked_borrows::Permission, AccessKind, GlobalStateInner, ProtectorKind,
};
use crate::*;

/// Error reporting
fn err_sb_ub<'tcx>(
    msg: String,
    help: Vec<String>,
    history: Option<TagHistory>,
) -> InterpError<'tcx> {
    err_machine_stop!(TerminationInfo::StackedBorrowsUb { msg, help, history })
}

#[derive(Clone, Debug)]
pub struct AllocHistory {
    id: AllocId,
    base: (Item, Span),
    creations: smallvec::SmallVec<[Creation; 1]>,
    invalidations: smallvec::SmallVec<[Invalidation; 1]>,
    protectors: smallvec::SmallVec<[Protection; 1]>,
}

#[derive(Clone, Debug)]
struct Creation {
    retag: RetagOp,
    span: Span,
}

impl Creation {
    fn generate_diagnostic(&self) -> (String, SpanData) {
        let tag = self.retag.new_tag;
        if let Some(perm) = self.retag.permission {
            (
                format!(
                    "{tag:?} was created by a {:?} retag at offsets {:?}",
                    perm, self.retag.range,
                ),
                self.span.data(),
            )
        } else {
            assert!(self.retag.range.size == Size::ZERO);
            (
                format!(
                    "{tag:?} would have been created here, but this is a zero-size retag ({:?}) so the tag in question does not exist anywhere",
                    self.retag.range,
                ),
                self.span.data(),
            )
        }
    }
}

#[derive(Clone, Debug)]
struct Invalidation {
    tag: BorTag,
    range: AllocRange,
    span: Span,
    cause: InvalidationCause,
}

#[derive(Clone, Debug)]
enum InvalidationCause {
    Access(AccessKind),
    Retag(Permission, RetagInfo),
}

impl Invalidation {
    fn generate_diagnostic(&self) -> (String, SpanData) {
        let message = if matches!(
            self.cause,
            InvalidationCause::Retag(_, RetagInfo { cause: RetagCause::FnEntry, .. })
        ) {
            // For a FnEntry retag, our Span points at the caller.
            // See `DiagnosticCx::log_invalidation`.
            format!(
                "{:?} was later invalidated at offsets {:?} by a {} inside this call",
                self.tag, self.range, self.cause
            )
        } else {
            format!(
                "{:?} was later invalidated at offsets {:?} by a {}",
                self.tag, self.range, self.cause
            )
        };
        (message, self.span.data())
    }
}

impl fmt::Display for InvalidationCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvalidationCause::Access(kind) => write!(f, "{kind}"),
            InvalidationCause::Retag(perm, info) =>
                write!(f, "{perm:?} {retag}", retag = info.summary()),
        }
    }
}

#[derive(Clone, Debug)]
struct Protection {
    tag: BorTag,
    span: Span,
}

#[derive(Clone)]
pub struct TagHistory {
    pub created: (String, SpanData),
    pub invalidated: Option<(String, SpanData)>,
    pub protected: Option<(String, SpanData)>,
}

pub struct DiagnosticCxBuilder<'ecx, 'mir, 'tcx> {
    operation: Operation,
    machine: &'ecx MiriMachine<'mir, 'tcx>,
}

pub struct DiagnosticCx<'history, 'ecx, 'mir, 'tcx> {
    operation: Operation,
    machine: &'ecx MiriMachine<'mir, 'tcx>,
    history: &'history mut AllocHistory,
    offset: Size,
}

impl<'ecx, 'mir, 'tcx> DiagnosticCxBuilder<'ecx, 'mir, 'tcx> {
    pub fn build<'history>(
        self,
        history: &'history mut AllocHistory,
        offset: Size,
    ) -> DiagnosticCx<'history, 'ecx, 'mir, 'tcx> {
        DiagnosticCx { operation: self.operation, machine: self.machine, history, offset }
    }

    pub fn retag(
        machine: &'ecx MiriMachine<'mir, 'tcx>,
        info: RetagInfo,
        new_tag: BorTag,
        orig_tag: ProvenanceExtra,
        range: AllocRange,
    ) -> Self {
        let operation =
            Operation::Retag(RetagOp { info, new_tag, orig_tag, range, permission: None });

        DiagnosticCxBuilder { machine, operation }
    }

    pub fn read(
        machine: &'ecx MiriMachine<'mir, 'tcx>,
        tag: ProvenanceExtra,
        range: AllocRange,
    ) -> Self {
        let operation = Operation::Access(AccessOp { kind: AccessKind::Read, tag, range });
        DiagnosticCxBuilder { machine, operation }
    }

    pub fn write(
        machine: &'ecx MiriMachine<'mir, 'tcx>,
        tag: ProvenanceExtra,
        range: AllocRange,
    ) -> Self {
        let operation = Operation::Access(AccessOp { kind: AccessKind::Write, tag, range });
        DiagnosticCxBuilder { machine, operation }
    }

    pub fn dealloc(machine: &'ecx MiriMachine<'mir, 'tcx>, tag: ProvenanceExtra) -> Self {
        let operation = Operation::Dealloc(DeallocOp { tag });
        DiagnosticCxBuilder { machine, operation }
    }
}

impl<'history, 'ecx, 'mir, 'tcx> DiagnosticCx<'history, 'ecx, 'mir, 'tcx> {
    pub fn unbuild(self) -> DiagnosticCxBuilder<'ecx, 'mir, 'tcx> {
        DiagnosticCxBuilder { machine: self.machine, operation: self.operation }
    }
}

#[derive(Debug, Clone)]
enum Operation {
    Retag(RetagOp),
    Access(AccessOp),
    Dealloc(DeallocOp),
}

#[derive(Debug, Clone)]
struct RetagOp {
    info: RetagInfo,
    new_tag: BorTag,
    orig_tag: ProvenanceExtra,
    range: AllocRange,
    permission: Option<Permission>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetagInfo {
    pub cause: RetagCause,
    pub in_field: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RetagCause {
    Normal,
    InPlaceFnPassing,
    FnEntry,
    TwoPhase,
}

#[derive(Debug, Clone)]
struct AccessOp {
    kind: AccessKind,
    tag: ProvenanceExtra,
    range: AllocRange,
}

#[derive(Debug, Clone)]
struct DeallocOp {
    tag: ProvenanceExtra,
}

impl AllocHistory {
    pub fn new(id: AllocId, item: Item, machine: &MiriMachine<'_, '_>) -> Self {
        Self {
            id,
            base: (item, machine.current_span()),
            creations: SmallVec::new(),
            invalidations: SmallVec::new(),
            protectors: SmallVec::new(),
        }
    }
}

impl<'history, 'ecx, 'mir, 'tcx> DiagnosticCx<'history, 'ecx, 'mir, 'tcx> {
    pub fn start_grant(&mut self, perm: Permission) {
        let Operation::Retag(op) = &mut self.operation else {
            unreachable!(
                "start_grant must only be called during a retag, this is: {:?}",
                self.operation
            )
        };
        op.permission = Some(perm);

        let last_creation = &mut self.history.creations.last_mut().unwrap();
        match last_creation.retag.permission {
            None => {
                last_creation.retag.permission = Some(perm);
            }
            Some(previous) =>
                if previous != perm {
                    // 'Split up' the creation event.
                    let previous_range = last_creation.retag.range;
                    last_creation.retag.range = alloc_range(previous_range.start, self.offset);
                    let mut new_event = last_creation.clone();
                    new_event.retag.range = alloc_range(self.offset, previous_range.end());
                    new_event.retag.permission = Some(perm);
                    self.history.creations.push(new_event);
                },
        }
    }

    pub fn log_creation(&mut self) {
        let Operation::Retag(op) = &self.operation else {
            unreachable!("log_creation must only be called during a retag")
        };
        self.history
            .creations
            .push(Creation { retag: op.clone(), span: self.machine.current_span() });
    }

    pub fn log_invalidation(&mut self, tag: BorTag) {
        let mut span = self.machine.current_span();
        let (range, cause) = match &self.operation {
            Operation::Retag(RetagOp { info, range, permission, .. }) => {
                if info.cause == RetagCause::FnEntry {
                    span = self.machine.caller_span();
                }
                (*range, InvalidationCause::Retag(permission.unwrap(), *info))
            }
            Operation::Access(AccessOp { kind, range, .. }) =>
                (*range, InvalidationCause::Access(*kind)),
            Operation::Dealloc(_) => {
                // This can be reached, but never be relevant later since the entire allocation is
                // gone now.
                return;
            }
        };
        self.history.invalidations.push(Invalidation { tag, range, span, cause });
    }

    pub fn log_protector(&mut self) {
        let Operation::Retag(op) = &self.operation else {
            unreachable!("Protectors can only be created during a retag")
        };
        self.history
            .protectors
            .push(Protection { tag: op.new_tag, span: self.machine.current_span() });
    }

    pub fn get_logs_relevant_to(
        &self,
        tag: BorTag,
        protector_tag: Option<BorTag>,
    ) -> Option<TagHistory> {
        let Some(created) = self
            .history
            .creations
            .iter()
            .rev()
            .find_map(|event| {
                // First, look for a Creation event where the tag and the offset matches. This
                // ensures that we pick the right Creation event when a retag isn't uniform due to
                // Freeze.
                let range = event.retag.range;
                if event.retag.new_tag == tag
                    && self.offset >= range.start
                    && self.offset < (range.start + range.size)
                {
                    Some(event.generate_diagnostic())
                } else {
                    None
                }
            })
            .or_else(|| {
                // If we didn't find anything with a matching offset, just return the event where
                // the tag was created. This branch is hit when we use a tag at an offset that
                // doesn't have the tag.
                self.history.creations.iter().rev().find_map(|event| {
                    if event.retag.new_tag == tag {
                        Some(event.generate_diagnostic())
                    } else {
                        None
                    }
                })
            })
            .or_else(|| {
                // If we didn't find a retag that created this tag, it might be the base tag of
                // this allocation.
                if self.history.base.0.tag() == tag {
                    Some((
                        format!(
                            "{tag:?} was created here, as the base tag for {:?}",
                            self.history.id
                        ),
                        self.history.base.1.data(),
                    ))
                } else {
                    None
                }
            })
        else {
            // But if we don't have a creation event, this is related to a wildcard, and there
            // is really nothing we can do to help.
            return None;
        };

        let invalidated = self.history.invalidations.iter().rev().find_map(|event| {
            if event.tag == tag { Some(event.generate_diagnostic()) } else { None }
        });

        let protected = protector_tag
            .and_then(|protector| {
                self.history.protectors.iter().find(|protection| protection.tag == protector)
            })
            .map(|protection| {
                let protected_tag = protection.tag;
                (format!("{protected_tag:?} is this argument"), protection.span.data())
            });

        Some(TagHistory { created, invalidated, protected })
    }

    /// Report a descriptive error when `new` could not be granted from `derived_from`.
    #[inline(never)] // This is only called on fatal code paths
    pub(super) fn grant_error(&self, stack: &Stack) -> InterpError<'tcx> {
        let Operation::Retag(op) = &self.operation else {
            unreachable!("grant_error should only be called during a retag")
        };
        let perm =
            op.permission.expect("`start_grant` must be called before calling `grant_error`");
        let action = format!(
            "trying to retag from {:?} for {:?} permission at {:?}[{:#x}]",
            op.orig_tag,
            perm,
            self.history.id,
            self.offset.bytes(),
        );
        let mut helps = vec![operation_summary(&op.info.summary(), self.history.id, op.range)];
        if op.info.in_field {
            helps.push(format!("errors for retagging in fields are fairly new; please reach out to us (e.g. at <https://rust-lang.zulipchat.com/#narrow/stream/269128-miri>) if you find this error troubling"));
        }
        err_sb_ub(
            format!("{action}{}", error_cause(stack, op.orig_tag)),
            helps,
            op.orig_tag.and_then(|orig_tag| self.get_logs_relevant_to(orig_tag, None)),
        )
    }

    /// Report a descriptive error when `access` is not permitted based on `tag`.
    #[inline(never)] // This is only called on fatal code paths
    pub(super) fn access_error(&self, stack: &Stack) -> InterpError<'tcx> {
        // Deallocation and retagging also do an access as part of their thing, so handle that here, too.
        let op = match &self.operation {
            Operation::Access(op) => op,
            Operation::Retag(_) => return self.grant_error(stack),
            Operation::Dealloc(_) => return self.dealloc_error(stack),
        };
        let action = format!(
            "attempting a {access} using {tag:?} at {alloc_id:?}[{offset:#x}]",
            access = op.kind,
            tag = op.tag,
            alloc_id = self.history.id,
            offset = self.offset.bytes(),
        );
        err_sb_ub(
            format!("{action}{}", error_cause(stack, op.tag)),
            vec![operation_summary("an access", self.history.id, op.range)],
            op.tag.and_then(|tag| self.get_logs_relevant_to(tag, None)),
        )
    }

    #[inline(never)] // This is only called on fatal code paths
    pub(super) fn protector_error(&self, item: &Item, kind: ProtectorKind) -> InterpError<'tcx> {
        let protected = match kind {
            ProtectorKind::WeakProtector => "weakly protected",
            ProtectorKind::StrongProtector => "strongly protected",
        };
        let call_id = self
            .machine
            .threads
            .all_stacks()
            .flatten()
            .map(|frame| {
                frame.extra.borrow_tracker.as_ref().expect("we should have borrow tracking data")
            })
            .find(|frame| frame.protected_tags.contains(&item.tag()))
            .map(|frame| frame.call_id)
            .unwrap(); // FIXME: Surely we should find something, but a panic seems wrong here?
        match self.operation {
            Operation::Dealloc(_) =>
                err_sb_ub(
                    format!("deallocating while item {item:?} is {protected} by call {call_id:?}",),
                    vec![],
                    None,
                ),
            Operation::Retag(RetagOp { orig_tag: tag, .. })
            | Operation::Access(AccessOp { tag, .. }) =>
                err_sb_ub(
                    format!(
                        "not granting access to tag {tag:?} because that would remove {item:?} which is {protected} because it is an argument of call {call_id:?}",
                    ),
                    vec![],
                    tag.and_then(|tag| self.get_logs_relevant_to(tag, Some(item.tag()))),
                ),
        }
    }

    #[inline(never)] // This is only called on fatal code paths
    pub fn dealloc_error(&self, stack: &Stack) -> InterpError<'tcx> {
        let Operation::Dealloc(op) = &self.operation else {
            unreachable!("dealloc_error should only be called during a deallocation")
        };
        err_sb_ub(
            format!(
                "attempting deallocation using {tag:?} at {alloc_id:?}{cause}",
                tag = op.tag,
                alloc_id = self.history.id,
                cause = error_cause(stack, op.tag),
            ),
            vec![],
            op.tag.and_then(|tag| self.get_logs_relevant_to(tag, None)),
        )
    }

    #[inline(never)]
    pub fn check_tracked_tag_popped(&self, item: &Item, global: &GlobalStateInner) {
        if !global.tracked_pointer_tags.contains(&item.tag()) {
            return;
        }
        let cause = match self.operation {
            Operation::Dealloc(_) => format!(" due to deallocation"),
            Operation::Access(AccessOp { kind, tag, .. }) =>
                format!(" due to {kind:?} access for {tag:?}"),
            Operation::Retag(RetagOp { orig_tag, permission, new_tag, .. }) => {
                let permission = permission
                    .expect("start_grant should set the current permission before popping a tag");
                format!(
                    " due to {permission:?} retag from {orig_tag:?} (that retag created {new_tag:?})"
                )
            }
        };

        self.machine.emit_diagnostic(NonHaltingDiagnostic::PoppedPointerTag(*item, cause));
    }
}

fn operation_summary(operation: &str, alloc_id: AllocId, alloc_range: AllocRange) -> String {
    format!("this error occurs as part of {operation} at {alloc_id:?}{alloc_range:?}")
}

fn error_cause(stack: &Stack, prov_extra: ProvenanceExtra) -> &'static str {
    if let ProvenanceExtra::Concrete(tag) = prov_extra {
        if (0..stack.len())
            .map(|i| stack.get(i).unwrap())
            .any(|item| item.tag() == tag && item.perm() != Permission::Disabled)
        {
            ", but that tag only grants SharedReadOnly permission for this location"
        } else {
            ", but that tag does not exist in the borrow stack for this location"
        }
    } else {
        ", but no exposed tags have suitable permission in the borrow stack for this location"
    }
}

impl RetagInfo {
    fn summary(&self) -> String {
        let mut s = match self.cause {
            RetagCause::Normal => "retag",
            RetagCause::FnEntry => "function-entry retag",
            RetagCause::InPlaceFnPassing => "in-place function argument/return passing protection",
            RetagCause::TwoPhase => "two-phase retag",
        }
        .to_string();
        if self.in_field {
            s.push_str(" (of a reference/box inside this compound value)");
        }
        s
    }
}
