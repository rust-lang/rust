use smallvec::SmallVec;

use rustc_middle::mir::interpret::{AllocId, AllocRange};
use rustc_span::{Span, SpanData};
use rustc_target::abi::Size;

use crate::helpers::{CurrentSpan, HexRange};
use crate::stacked_borrows::{err_sb_ub, AccessKind, Permission};
use crate::Item;
use crate::SbTag;
use crate::SbTagExtra;
use crate::Stack;

use rustc_middle::mir::interpret::InterpError;

#[derive(Clone, Debug)]
pub struct AllocHistory {
    // The time tags can be compressed down to one bit per event, by just storing a Vec<u8>
    // where each bit is set to indicate if the event was a creation or a retag
    current_time: usize,
    creations: smallvec::SmallVec<[Event; 2]>,
    invalidations: smallvec::SmallVec<[Event; 1]>,
    protectors: smallvec::SmallVec<[Protection; 1]>,
}

#[derive(Clone, Debug)]
struct Protection {
    orig_tag: SbTag,
    tag: SbTag,
    span: Span,
}

#[derive(Clone, Debug)]
struct Event {
    parent: Option<SbTag>,
    tag: SbTag,
    range: AllocRange,
    span: Span,
}

pub enum TagHistory {
    Tagged {
        tag: SbTag,
        created: (AllocRange, SpanData),
        invalidated: Option<(AllocRange, SpanData)>,
        protected: Option<(SbTag, SpanData, SpanData)>,
    },
}

impl AllocHistory {
    pub fn new() -> Self {
        Self {
            current_time: 0,
            creations: SmallVec::new(),
            invalidations: SmallVec::new(),
            protectors: SmallVec::new(),
        }
    }

    pub fn log_creation(
        &mut self,
        parent: Option<SbTag>,
        tag: SbTag,
        range: AllocRange,
        current_span: &mut CurrentSpan<'_, '_, '_>,
    ) {
        let span = current_span.get();
        self.creations.push(Event { parent, tag, range, span });
        self.current_time += 1;
    }

    pub fn log_invalidation(
        &mut self,
        tag: SbTag,
        range: AllocRange,
        current_span: &mut CurrentSpan<'_, '_, '_>,
    ) {
        let span = current_span.get();
        self.invalidations.push(Event { parent: None, tag, range, span });
        self.current_time += 1;
    }

    pub fn log_protector(
        &mut self,
        orig_tag: SbTag,
        tag: SbTag,
        current_span: &mut CurrentSpan<'_, '_, '_>,
    ) {
        let span = current_span.get();
        self.protectors.push(Protection { orig_tag, tag, span });
        self.current_time += 1;
    }

    pub fn get_logs_relevant_to(
        &self,
        tag: SbTag,
        protector_tag: Option<SbTag>,
    ) -> Option<TagHistory> {
        let protected = protector_tag
            .and_then(|protector| {
                self.protectors.iter().find_map(|protection| {
                    if protection.tag == protector {
                        Some((protection.orig_tag, protection.span.data()))
                    } else {
                        None
                    }
                })
            })
            .and_then(|(tag, call_span)| {
                self.creations.iter().rev().find_map(|event| {
                    if event.tag == tag {
                        Some((event.parent?, event.span.data(), call_span))
                    } else {
                        None
                    }
                })
            });

        let get_matching = |events: &[Event]| {
            events.iter().rev().find_map(|event| {
                if event.tag == tag { Some((event.range, event.span.data())) } else { None }
            })
        };
        Some(TagHistory::Tagged {
            tag,
            created: get_matching(&self.creations)?,
            invalidated: get_matching(&self.invalidations),
            protected,
        })
    }

    /// Report a descriptive error when `new` could not be granted from `derived_from`.
    pub fn grant_error<'tcx>(
        &self,
        derived_from: SbTagExtra,
        new: Item,
        alloc_id: AllocId,
        alloc_range: AllocRange,
        error_offset: Size,
        stack: &Stack,
    ) -> InterpError<'tcx> {
        let action = format!(
            "trying to reborrow {derived_from:?} for {:?} permission at {alloc_id:?}[{:#x}]",
            new.perm,
            error_offset.bytes(),
        );
        err_sb_ub(
            format!("{}{}", action, error_cause(stack, derived_from)),
            Some(operation_summary("a reborrow", alloc_id, alloc_range)),
            derived_from.and_then(|derived_from| self.get_logs_relevant_to(derived_from, None)),
        )
    }

    /// Report a descriptive error when `access` is not permitted based on `tag`.
    pub fn access_error<'tcx>(
        &self,
        access: AccessKind,
        tag: SbTagExtra,
        alloc_id: AllocId,
        alloc_range: AllocRange,
        error_offset: Size,
        stack: &Stack,
    ) -> InterpError<'tcx> {
        let action = format!(
            "attempting a {access} using {tag:?} at {alloc_id:?}[{:#x}]",
            error_offset.bytes(),
        );
        err_sb_ub(
            format!("{}{}", action, error_cause(stack, tag)),
            Some(operation_summary("an access", alloc_id, alloc_range)),
            tag.and_then(|tag| self.get_logs_relevant_to(tag, None)),
        )
    }
}

fn operation_summary(
    operation: &'static str,
    alloc_id: AllocId,
    alloc_range: AllocRange,
) -> String {
    format!("this error occurs as part of {} at {:?}{}", operation, alloc_id, HexRange(alloc_range))
}

fn error_cause(stack: &Stack, tag: SbTagExtra) -> &'static str {
    if let SbTagExtra::Concrete(tag) = tag {
        if (0..stack.len())
            .map(|i| stack.get(i).unwrap())
            .any(|item| item.tag == tag && item.perm != Permission::Disabled)
        {
            ", but that tag only grants SharedReadOnly permission for this location"
        } else {
            ", but that tag does not exist in the borrow stack for this location"
        }
    } else {
        ", but no exposed tags have suitable permission in the borrow stack for this location"
    }
}
