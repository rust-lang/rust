use rustc_middle::mir::interpret::{AllocId, AllocRange};
use rustc_span::{Span, SpanData};
use rustc_target::abi::Size;

use crate::helpers::HexRange;
use crate::stacked_borrows::{err_sb_ub, AccessKind, GlobalStateInner, Permission};
use crate::Item;
use crate::SbTag;
use crate::Stack;

use rustc_middle::mir::interpret::InterpError;

#[derive(Debug, Default)]
pub struct AllocHistory {
    // The time tags can be compressed down to one bit per event, by just storing a Vec<u8>
    // where each bit is set to indicate if the event was a creation or a retag
    current_time: usize,
    creations: smallvec::SmallVec<[Event; 2]>,
    invalidations: smallvec::SmallVec<[Event; 1]>,
    protectors: smallvec::SmallVec<[Protection; 1]>,
}

#[derive(Debug)]
struct Protection {
    orig_tag: SbTag,
    tag: SbTag,
    span: Span,
}

#[derive(Debug)]
struct Event {
    time: usize,
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
    Untagged {
        recently_created: Option<(AllocRange, SpanData)>,
        recently_invalidated: Option<(AllocRange, SpanData)>,
        matching_created: Option<(AllocRange, SpanData)>,
        protected: Option<(SbTag, SpanData, SpanData)>,
    },
}

pub trait GlobalStateExt {
    fn add_creation(
        &mut self,
        parent: Option<SbTag>,
        tag: SbTag,
        alloc: AllocId,
        range: AllocRange,
    );

    fn add_invalidation(&mut self, tag: SbTag, alloc: AllocId, range: AllocRange);

    fn add_protector(&mut self, orig_tag: SbTag, tag: SbTag, alloc: AllocId);

    fn get_stack_history(
        &self,
        tag: SbTag,
        alloc: AllocId,
        alloc_range: AllocRange,
        offset: Size,
        protector_tag: Option<SbTag>,
    ) -> Option<TagHistory>;
}

impl GlobalStateExt for GlobalStateInner {
    fn add_creation(
        &mut self,
        parent: Option<SbTag>,
        tag: SbTag,
        alloc: AllocId,
        range: AllocRange,
    ) {
        let extras = self.extras.entry(alloc).or_default();
        extras.creations.push(Event {
            parent,
            tag,
            range,
            span: self.current_span,
            time: extras.current_time,
        });
        extras.current_time += 1;
    }

    fn add_invalidation(&mut self, tag: SbTag, alloc: AllocId, range: AllocRange) {
        let extras = self.extras.entry(alloc).or_default();
        extras.invalidations.push(Event {
            parent: None,
            tag,
            range,
            span: self.current_span,
            time: extras.current_time,
        });
        extras.current_time += 1;
    }

    fn add_protector(&mut self, orig_tag: SbTag, tag: SbTag, alloc: AllocId) {
        let extras = self.extras.entry(alloc).or_default();
        extras.protectors.push(Protection { orig_tag, tag, span: self.current_span });
        extras.current_time += 1;
    }

    fn get_stack_history(
        &self,
        tag: SbTag,
        alloc: AllocId,
        alloc_range: AllocRange,
        offset: Size,
        protector_tag: Option<SbTag>,
    ) -> Option<TagHistory> {
        let extras = self.extras.get(&alloc)?;
        let protected = protector_tag
            .and_then(|protector| {
                extras.protectors.iter().find_map(|protection| {
                    if protection.tag == protector {
                        Some((protection.orig_tag, protection.span.data()))
                    } else {
                        None
                    }
                })
            })
            .and_then(|(tag, call_span)| {
                extras.creations.iter().rev().find_map(|event| {
                    if event.tag == tag {
                        Some((event.parent?, event.span.data(), call_span))
                    } else {
                        None
                    }
                })
            });
        if let SbTag::Tagged(_) = tag {
            let get_matching = |events: &[Event]| {
                events.iter().rev().find_map(|event| {
                    if event.tag == tag { Some((event.range, event.span.data())) } else { None }
                })
            };
            Some(TagHistory::Tagged {
                tag,
                created: get_matching(&extras.creations)?,
                invalidated: get_matching(&extras.invalidations),
                protected,
            })
        } else {
            let mut created_time = 0;
            // Find the most recently created tag that satsfies this offset
            let recently_created = extras.creations.iter().rev().find_map(|event| {
                if event.tag == tag && offset >= event.range.start && offset < event.range.end() {
                    created_time = event.time;
                    Some((event.range, event.span.data()))
                } else {
                    None
                }
            });

            // Find a different recently created tag that satisfies this whole operation, predates
            // the recently created tag, and has a different span.
            // We're trying to make a guess at which span the user wanted to provide the tag that
            // they're using.
            let matching_created = if let Some((_created_range, created_span)) = recently_created {
                extras.creations.iter().rev().find_map(|event| {
                    if event.tag == tag
                        && alloc_range.start >= event.range.start
                        && alloc_range.end() <= event.range.end()
                        && event.span.data() != created_span
                        && event.time != created_time
                    {
                        Some((event.range, event.span.data()))
                    } else {
                        None
                    }
                })
            } else {
                None
            };

            let recently_invalidated = if recently_created.is_some() {
                // Find the most recent invalidation of this tag which post-dates the creation
                let mut found = None;
                for event in extras.invalidations.iter().rev() {
                    if event.time < created_time {
                        break;
                    }
                    if event.tag == tag && offset >= event.range.start && offset < event.range.end()
                    {
                        found = Some((event.range, event.span.data()))
                    }
                }
                found
            } else {
                None
            };
            Some(TagHistory::Untagged {
                recently_created,
                matching_created,
                recently_invalidated,
                protected,
            })
        }
    }
}

pub trait StackExt {
    fn grant_error(
        &self,
        derived_from: SbTag,
        new: Item,
        alloc_id: AllocId,
        alloc_range: AllocRange,
        error_offset: Size,
        global: &GlobalStateInner,
    ) -> InterpError<'static>;

    fn access_error(
        &self,
        access: AccessKind,
        tag: SbTag,
        alloc_id: AllocId,
        alloc_range: AllocRange,
        error_offset: Size,
        global: &GlobalStateInner,
    ) -> InterpError<'static>;
}

impl StackExt for Stack {
    /// Report a descriptive error when `new` could not be granted from `derived_from`.
    fn grant_error(
        &self,
        derived_from: SbTag,
        new: Item,
        alloc_id: AllocId,
        alloc_range: AllocRange,
        error_offset: Size,
        global: &GlobalStateInner,
    ) -> InterpError<'static> {
        let action = format!(
            "trying to reborrow {:?} for {:?} permission at {}[{:#x}]",
            derived_from,
            new.perm,
            alloc_id,
            error_offset.bytes(),
        );
        err_sb_ub(
            format!("{}{}", action, error_cause(self, derived_from)),
            Some(operation_summary("a reborrow", alloc_id, alloc_range)),
            global.get_stack_history(derived_from, alloc_id, alloc_range, error_offset, None),
        )
    }

    /// Report a descriptive error when `access` is not permitted based on `tag`.
    fn access_error(
        &self,
        access: AccessKind,
        tag: SbTag,
        alloc_id: AllocId,
        alloc_range: AllocRange,
        error_offset: Size,
        global: &GlobalStateInner,
    ) -> InterpError<'static> {
        let action = format!(
            "attempting a {} using {:?} at {}[{:#x}]",
            access,
            tag,
            alloc_id,
            error_offset.bytes(),
        );
        err_sb_ub(
            format!("{}{}", action, error_cause(self, tag)),
            Some(operation_summary("an access", alloc_id, alloc_range)),
            global.get_stack_history(tag, alloc_id, alloc_range, error_offset, None),
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

fn error_cause(stack: &Stack, tag: SbTag) -> &'static str {
    if stack.borrows.iter().any(|item| item.tag == tag && item.perm != Permission::Disabled) {
        ", but that tag only grants SharedReadOnly permission for this location"
    } else {
        ", but that tag does not exist in the borrow stack for this location"
    }
}
