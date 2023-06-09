use std::fmt;
use std::ops::Range;

use rustc_data_structures::fx::FxHashMap;
use rustc_span::{Span, SpanData};

use crate::borrow_tracker::tree_borrows::{
    perms::{PermTransition, Permission},
    tree::LocationState,
    unimap::UniIndex,
};
use crate::borrow_tracker::{AccessKind, ProtectorKind};
use crate::*;

/// Complete data for an event:
#[derive(Clone, Debug)]
pub struct Event {
    /// Transformation of permissions that occured because of this event
    pub transition: PermTransition,
    /// Kind of the access that triggered this event
    pub access_kind: AccessKind,
    /// Relative position of the tag to the one used for the access
    pub is_foreign: bool,
    /// User-visible range of the access
    pub access_range: AllocRange,
    /// The transition recorded by this event only occured on a subrange of
    /// `access_range`: a single access on `access_range` triggers several events,
    /// each with their own mutually disjoint `transition_range`. No-op transitions
    /// should not be recorded as events, so the union of all `transition_range` is not
    /// necessarily the entire `access_range`.
    ///
    /// No data from any `transition_range` should ever be user-visible, because
    /// both the start and end of `transition_range` are entirely dependent on the
    /// internal representation of `RangeMap` which is supposed to be opaque.
    /// What will be shown in the error message is the first byte `error_offset` of
    /// the `TbError`, which should satisfy
    /// `event.transition_range.contains(error.error_offset)`.
    pub transition_range: Range<u64>,
    /// Line of code that triggered this event
    pub span: Span,
}

/// List of all events that affected a tag.
/// NOTE: not all of these events are relevant for a particular location,
/// the events should be filtered before the generation of diagnostics.
/// Available filtering methods include `History::forget` and `History::extract_relevant`.
#[derive(Clone, Debug)]
pub struct History {
    tag: BorTag,
    created: (Span, Permission),
    events: Vec<Event>,
}

/// History formatted for use by `src/diagnostics.rs`.
///
/// NOTE: needs to be `Send` because of a bound on `MachineStopType`, hence
/// the use of `SpanData` rather than `Span`.
#[derive(Debug, Clone, Default)]
pub struct HistoryData {
    pub events: Vec<(Option<SpanData>, String)>, // includes creation
}

impl History {
    /// Record an additional event to the history.
    pub fn push(&mut self, event: Event) {
        self.events.push(event);
    }
}

impl HistoryData {
    // Format events from `new_history` into those recorded by `self`.
    //
    // NOTE: also converts `Span` to `SpanData`.
    fn extend(&mut self, new_history: History, tag_name: &'static str, show_initial_state: bool) {
        let History { tag, created, events } = new_history;
        let this = format!("the {tag_name} tag {tag:?}");
        let msg_initial_state = format!(", in the initial state {}", created.1);
        let msg_creation = format!(
            "{this} was created here{maybe_msg_initial_state}",
            maybe_msg_initial_state = if show_initial_state { &msg_initial_state } else { "" },
        );

        self.events.push((Some(created.0.data()), msg_creation));
        for &Event {
            transition,
            access_kind,
            is_foreign,
            access_range,
            span,
            transition_range: _,
        } in &events
        {
            // NOTE: `transition_range` is explicitly absent from the error message, it has no significance
            // to the user. The meaningful one is `access_range`.
            self.events.push((Some(span.data()), format!("{this} later transitioned to {endpoint} due to a {rel} {access_kind} at offsets {access_range:?}", endpoint = transition.endpoint(), rel = if is_foreign { "foreign" } else { "child" })));
            self.events.push((None, format!("this corresponds to {}", transition.summary())));
        }
    }
}

/// Some information that is irrelevant for the algorithm but very
/// convenient to know about a tag for debugging and testing.
#[derive(Clone, Debug)]
pub struct NodeDebugInfo {
    /// The tag in question.
    pub tag: BorTag,
    /// Name(s) that were associated with this tag (comma-separated).
    /// Typically the name of the variable holding the corresponding
    /// pointer in the source code.
    /// Helps match tag numbers to human-readable names.
    pub name: Option<String>,
    /// Notable events in the history of this tag, used for
    /// diagnostics.
    ///
    /// NOTE: by virtue of being part of `NodeDebugInfo`,
    /// the history is automatically cleaned up by the GC.
    /// NOTE: this is `!Send`, it needs to be converted before displaying
    /// the actual diagnostics because `src/diagnostics.rs` requires `Send`.
    pub history: History,
}

impl NodeDebugInfo {
    /// Information for a new node. By default it has no
    /// name and an empty history.
    pub fn new(tag: BorTag, initial: Permission, span: Span) -> Self {
        let history = History { tag, created: (span, initial), events: Vec::new() };
        Self { tag, name: None, history }
    }

    /// Add a name to the tag. If a same tag is associated to several pointers,
    /// it can have several names which will be separated by commas.
    pub fn add_name(&mut self, name: &str) {
        if let Some(ref mut prev_name) = &mut self.name {
            prev_name.push_str(", ");
            prev_name.push_str(name);
        } else {
            self.name = Some(String::from(name));
        }
    }
}

impl fmt::Display for NodeDebugInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref name) = self.name {
            write!(f, "{tag:?} ({name})", tag = self.tag)
        } else {
            write!(f, "{tag:?}", tag = self.tag)
        }
    }
}

impl<'tcx> Tree {
    /// Climb the tree to get the tag of a distant ancestor.
    /// Allows operations on tags that are unreachable by the program
    /// but still exist in the tree. Not guaranteed to perform consistently
    /// if `tag-gc=1`.
    fn nth_parent(&self, tag: BorTag, nth_parent: u8) -> Option<BorTag> {
        let mut idx = self.tag_mapping.get(&tag).unwrap();
        for _ in 0..nth_parent {
            let node = self.nodes.get(idx).unwrap();
            idx = node.parent?;
        }
        Some(self.nodes.get(idx).unwrap().tag)
    }

    /// Debug helper: assign name to tag.
    pub fn give_pointer_debug_name(
        &mut self,
        tag: BorTag,
        nth_parent: u8,
        name: &str,
    ) -> InterpResult<'tcx> {
        let tag = self.nth_parent(tag, nth_parent).unwrap();
        let idx = self.tag_mapping.get(&tag).unwrap();
        if let Some(node) = self.nodes.get_mut(idx) {
            node.debug_info.add_name(name);
        } else {
            eprintln!("Tag {tag:?} (to be named '{name}') not found!");
        }
        Ok(())
    }

    /// Debug helper: determines if the tree contains a tag.
    pub fn is_allocation_of(&self, tag: BorTag) -> bool {
        self.tag_mapping.contains_key(&tag)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum TransitionError {
    /// This access is not allowed because some parent tag has insufficient permissions.
    /// For example, if a tag is `Frozen` and encounters a child write this will
    /// produce a `ChildAccessForbidden(Frozen)`.
    /// This kind of error can only occur on child accesses.
    ChildAccessForbidden(Permission),
    /// A protector was triggered due to an invalid transition that loses
    /// too much permissions.
    /// For example, if a protected tag goes from `Active` to `Frozen` due
    /// to a foreign write this will produce a `ProtectedTransition(PermTransition(Active, Frozen))`.
    /// This kind of error can only occur on foreign accesses.
    ProtectedTransition(PermTransition),
    /// Cannot deallocate because some tag in the allocation is strongly protected.
    /// This kind of error can only occur on deallocations.
    ProtectedDealloc,
}

impl History {
    /// Keep only the tag and creation
    fn forget(&self) -> Self {
        History { events: Vec::new(), created: self.created, tag: self.tag }
    }

    /// Reconstruct the history relevant to `error_offset` by filtering
    /// only events whose range contains the offset we are interested in.
    fn extract_relevant(&self, error_offset: u64, error_kind: TransitionError) -> Self {
        History {
            events: self
                .events
                .iter()
                .filter(|e| e.transition_range.contains(&error_offset))
                .filter(|e| e.transition.is_relevant(error_kind))
                .cloned()
                .collect::<Vec<_>>(),
            created: self.created,
            tag: self.tag,
        }
    }
}

/// Failures that can occur during the execution of Tree Borrows procedures.
pub(super) struct TbError<'node> {
    /// What failure occurred.
    pub error_kind: TransitionError,
    /// The offset (into the allocation) at which the conflict occurred.
    pub error_offset: u64,
    /// The tag on which the error was triggered.
    /// On protector violations, this is the tag that was protected.
    /// On accesses rejected due to insufficient permissions, this is the
    /// tag that lacked those permissions.
    pub conflicting_info: &'node NodeDebugInfo,
    /// Whether this was a Read or Write access. This field is ignored
    /// when the error was triggered by a deallocation.
    pub access_kind: AccessKind,
    /// Which tag the access that caused this error was made through, i.e.
    /// which tag was used to read/write/deallocate.
    pub accessed_info: &'node NodeDebugInfo,
}

impl TbError<'_> {
    /// Produce a UB error.
    pub fn build<'tcx>(self) -> InterpError<'tcx> {
        use TransitionError::*;
        let kind = self.access_kind;
        let accessed = self.accessed_info;
        let conflicting = self.conflicting_info;
        let accessed_is_conflicting = accessed.tag == conflicting.tag;
        let (title, details, conflicting_tag_name) = match self.error_kind {
            ChildAccessForbidden(perm) => {
                let conflicting_tag_name =
                    if accessed_is_conflicting { "accessed" } else { "conflicting" };
                let title = format!("{kind} through {accessed} is forbidden");
                let mut details = Vec::new();
                if !accessed_is_conflicting {
                    details.push(format!(
                        "the accessed tag {accessed} is a child of the conflicting tag {conflicting}"
                    ));
                }
                details.push(format!(
                    "the {conflicting_tag_name} tag {conflicting} has state {perm} which forbids child {kind}es"
                ));
                (title, details, conflicting_tag_name)
            }
            ProtectedTransition(transition) => {
                let conflicting_tag_name = "protected";
                let title = format!("{kind} through {accessed} is forbidden");
                let details = vec![
                    format!(
                        "the accessed tag {accessed} is foreign to the {conflicting_tag_name} tag {conflicting} (i.e., it is not a child)"
                    ),
                    format!(
                        "the access would cause the {conflicting_tag_name} tag {conflicting} to transition {transition}"
                    ),
                    format!(
                        "this is {loss}, which is not allowed for protected tags",
                        loss = transition.summary(),
                    ),
                ];
                (title, details, conflicting_tag_name)
            }
            ProtectedDealloc => {
                let conflicting_tag_name = "strongly protected";
                let title = format!("deallocation through {accessed} is forbidden");
                let details = vec![
                    format!(
                        "the allocation of the accessed tag {accessed} also contains the {conflicting_tag_name} tag {conflicting}"
                    ),
                    format!("the {conflicting_tag_name} tag {conflicting} disallows deallocations"),
                ];
                (title, details, conflicting_tag_name)
            }
        };
        let mut history = HistoryData::default();
        if !accessed_is_conflicting {
            history.extend(self.accessed_info.history.forget(), "accessed", false);
        }
        history.extend(
            self.conflicting_info.history.extract_relevant(self.error_offset, self.error_kind),
            conflicting_tag_name,
            true,
        );
        err_machine_stop!(TerminationInfo::TreeBorrowsUb { title, details, history })
    }
}

type S = &'static str;
/// Pretty-printing details
///
/// Example:
/// ```
/// DisplayFmtWrapper {
///     top: '>',
///     bot: '<',
///     warning_text: "Some tags have been hidden",
/// }
/// ```
/// will wrap the entire text with
/// ```text
/// >>>>>>>>>>>>>>>>>>>>>>>>>>
/// Some tags have been hidden
///
/// [ main display here ]
///
/// <<<<<<<<<<<<<<<<<<<<<<<<<<
/// ```
struct DisplayFmtWrapper {
    /// Character repeated to make the upper border.
    top: char,
    /// Character repeated to make the lower border.
    bot: char,
    /// Warning about some tags (unnamed) being hidden.
    warning_text: S,
}

/// Formating of the permissions on each range.
///
/// Example:
/// ```
/// DisplayFmtPermission {
///     open: "[",
///     sep: "|",
///     close: "]",
///     uninit: "___",
///     range_sep: "..",
/// }
/// ```
/// will show each permission line as
/// ```text
/// 0.. 1.. 2.. 3.. 4.. 5
/// [Act|Res|Frz|Dis|___]
/// ```
struct DisplayFmtPermission {
    /// Text that starts the permission block.
    open: S,
    /// Text that separates permissions on different ranges.
    sep: S,
    /// Text that ends the permission block.
    close: S,
    /// Text to show when a permission is not initialized.
    /// Should have the same width as a `Permission`'s `.short_name()`, i.e.
    /// 3 if using the `Res/Act/Frz/Dis` notation.
    uninit: S,
    /// Text to separate the `start` and `end` values of a range.
    range_sep: S,
}

/// Formating of the tree structure.
///
/// Example:
/// ```
/// DisplayFmtPadding {
///     join_middle: "|-",
///     join_last: "'-",
///     join_haschild: "-+-",
///     join_default: "---",
///     indent_middle: "| ",
///     indent_last: "  ",
/// }
/// ```
/// will show the tree as
/// ```text
/// -+- root
///  |--+- a
///  |  '--+- b
///  |     '---- c
///  |--+- d
///  |  '---- e
///  '---- f
/// ```
struct DisplayFmtPadding {
    /// Connector for a child other than the last.
    join_middle: S,
    /// Connector for the last child. Should have the same width as `join_middle`.
    join_last: S,
    /// Connector for a node that itself has a child.
    join_haschild: S,
    /// Connector for a node that does not have a child. Should have the same width
    /// as `join_haschild`.
    join_default: S,
    /// Indentation when there is a next child.
    indent_middle: S,
    /// Indentation for the last child.
    indent_last: S,
}
/// How to show whether a location has been accessed
///
/// Example:
/// ```
/// DisplayFmtAccess {
///     yes: " ",
///     no: "?",
///     meh: "_",
/// }
/// ```
/// will show states as
/// ```text
///  Act
/// ?Res
/// ____
/// ```
struct DisplayFmtAccess {
    /// Used when `State.initialized = true`.
    yes: S,
    /// Used when `State.initialized = false`.
    /// Should have the same width as `yes`.
    no: S,
    /// Used when there is no `State`.
    /// Should have the same width as `yes`.
    meh: S,
}

/// All parameters to determine how the tree is formated.
struct DisplayFmt {
    wrapper: DisplayFmtWrapper,
    perm: DisplayFmtPermission,
    padding: DisplayFmtPadding,
    accessed: DisplayFmtAccess,
}
impl DisplayFmt {
    /// Print the permission with the format
    /// ` Res`/` Re*`/` Act`/` Frz`/` Dis` for accessed locations
    /// and `?Res`/`?Re*`/`?Act`/`?Frz`/`?Dis` for unaccessed locations.
    fn print_perm(&self, perm: Option<LocationState>) -> String {
        if let Some(perm) = perm {
            format!(
                "{ac}{st}",
                ac = if perm.is_initialized() { self.accessed.yes } else { self.accessed.no },
                st = perm.permission().short_name(),
            )
        } else {
            format!("{}{}", self.accessed.meh, self.perm.uninit)
        }
    }

    /// Print the tag with the format `<XYZ>` if the tag is unnamed,
    /// and `<XYZ=name>` if the tag is named.
    fn print_tag(&self, tag: BorTag, name: &Option<String>) -> String {
        let printable_tag = tag.get();
        if let Some(name) = name {
            format!("<{printable_tag}={name}>")
        } else {
            format!("<{printable_tag}>")
        }
    }

    /// Print extra text if the tag has a protector.
    fn print_protector(&self, protector: Option<&ProtectorKind>) -> &'static str {
        protector
            .map(|p| {
                match *p {
                    ProtectorKind::WeakProtector => " Weakly protected",
                    ProtectorKind::StrongProtector => " Strongly protected",
                }
            })
            .unwrap_or("")
    }
}

/// Track the indentation of the tree.
struct DisplayIndent {
    curr: String,
}
impl DisplayIndent {
    fn new() -> Self {
        Self { curr: "    ".to_string() }
    }

    /// Increment the indentation by one. Note: need to know if this
    /// is the last child or not because the presence of other children
    /// changes the way the indentation is shown.
    fn increment(&mut self, formatter: &DisplayFmt, is_last: bool) {
        self.curr.push_str(if is_last {
            formatter.padding.indent_last
        } else {
            formatter.padding.indent_middle
        });
    }

    /// Pop the last level of indentation.
    fn decrement(&mut self, formatter: &DisplayFmt) {
        for _ in 0..formatter.padding.indent_last.len() {
            let _ = self.curr.pop();
        }
    }

    /// Print the current indentation.
    fn write(&self, s: &mut String) {
        s.push_str(&self.curr);
    }
}

/// Repeat a character a number of times.
fn char_repeat(c: char, n: usize) -> String {
    std::iter::once(c).cycle().take(n).collect::<String>()
}

/// Extracted information from the tree, in a form that is readily accessible
/// for printing. I.e. resolve parent-child pointers into an actual tree,
/// zip permissions with their tag, remove wrappers, stringify data.
struct DisplayRepr {
    tag: BorTag,
    name: Option<String>,
    rperm: Vec<Option<LocationState>>,
    children: Vec<DisplayRepr>,
}

impl DisplayRepr {
    fn from(tree: &Tree, show_unnamed: bool) -> Option<Self> {
        let mut v = Vec::new();
        extraction_aux(tree, tree.root, show_unnamed, &mut v);
        let Some(root) = v.pop() else {
            if show_unnamed {
                unreachable!("This allocation contains no tags, not even a root. This should not happen.");
            }
            eprintln!("This allocation does not contain named tags. Use `miri_print_borrow_state(_, true)` to also print unnamed tags.");
            return None;
        };
        assert!(v.is_empty());
        return Some(root);

        fn extraction_aux(
            tree: &Tree,
            idx: UniIndex,
            show_unnamed: bool,
            acc: &mut Vec<DisplayRepr>,
        ) {
            let node = tree.nodes.get(idx).unwrap();
            let name = node.debug_info.name.clone();
            let children_sorted = {
                let mut children = node.children.iter().cloned().collect::<Vec<_>>();
                children.sort_by_key(|idx| tree.nodes.get(*idx).unwrap().tag);
                children
            };
            if !show_unnamed && name.is_none() {
                // We skip this node
                for child_idx in children_sorted {
                    extraction_aux(tree, child_idx, show_unnamed, acc);
                }
            } else {
                // We take this node
                let rperm = tree
                    .rperms
                    .iter_all()
                    .map(move |(_offset, perms)| {
                        let perm = perms.get(idx);
                        perm.cloned()
                    })
                    .collect::<Vec<_>>();
                let mut children = Vec::new();
                for child_idx in children_sorted {
                    extraction_aux(tree, child_idx, show_unnamed, &mut children);
                }
                acc.push(DisplayRepr { tag: node.tag, name, rperm, children });
            }
        }
    }
    fn print(
        &self,
        fmt: &DisplayFmt,
        indenter: &mut DisplayIndent,
        protected_tags: &FxHashMap<BorTag, ProtectorKind>,
        ranges: Vec<Range<u64>>,
        print_warning: bool,
    ) {
        let mut block = Vec::new();
        // Push the header and compute the required paddings for the body.
        // Header looks like this: `0.. 1.. 2.. 3.. 4.. 5.. 6.. 7.. 8`,
        // and is properly aligned with the `|` of the body.
        let (range_header, range_padding) = {
            let mut header_top = String::new();
            header_top.push_str("0..");
            let mut padding = Vec::new();
            for (i, range) in ranges.iter().enumerate() {
                if i > 0 {
                    header_top.push_str(fmt.perm.range_sep);
                }
                let s = range.end.to_string();
                let l = s.chars().count() + fmt.perm.range_sep.chars().count();
                {
                    let target_len =
                        fmt.perm.uninit.chars().count() + fmt.accessed.yes.chars().count() + 1;
                    let tot_len = target_len.max(l);
                    let header_top_pad_len = target_len.saturating_sub(l);
                    let body_pad_len = tot_len.saturating_sub(target_len);
                    header_top.push_str(&format!("{}{}", char_repeat(' ', header_top_pad_len), s));
                    padding.push(body_pad_len);
                }
            }
            ([header_top], padding)
        };
        for s in range_header {
            block.push(s);
        }
        // This is the actual work
        print_aux(
            self,
            &range_padding,
            fmt,
            indenter,
            protected_tags,
            true, /* root _is_ the last child */
            &mut block,
        );
        // Then it's just prettifying it with a border of dashes.
        {
            let wr = &fmt.wrapper;
            let max_width = {
                let block_width = block.iter().map(|s| s.chars().count()).max().unwrap();
                if print_warning {
                    block_width.max(wr.warning_text.chars().count())
                } else {
                    block_width
                }
            };
            eprintln!("{}", char_repeat(wr.top, max_width));
            if print_warning {
                eprintln!("{}", wr.warning_text,);
            }
            for line in block {
                eprintln!("{line}");
            }
            eprintln!("{}", char_repeat(wr.bot, max_width));
        }

        // Here is the function that does the heavy lifting
        fn print_aux(
            tree: &DisplayRepr,
            padding: &[usize],
            fmt: &DisplayFmt,
            indent: &mut DisplayIndent,
            protected_tags: &FxHashMap<BorTag, ProtectorKind>,
            is_last_child: bool,
            acc: &mut Vec<String>,
        ) {
            let mut line = String::new();
            // Format the permissions on each range.
            // Looks like `| Act| Res| Res| Act|`.
            line.push_str(fmt.perm.open);
            for (i, (perm, &pad)) in tree.rperm.iter().zip(padding.iter()).enumerate() {
                if i > 0 {
                    line.push_str(fmt.perm.sep);
                }
                let show_perm = fmt.print_perm(*perm);
                line.push_str(&format!("{}{}", char_repeat(' ', pad), show_perm));
            }
            line.push_str(fmt.perm.close);
            // Format the tree structure.
            // Main difficulty is handling the indentation properly.
            indent.write(&mut line);
            {
                // padding
                line.push_str(if is_last_child {
                    fmt.padding.join_last
                } else {
                    fmt.padding.join_middle
                });
                line.push_str(fmt.padding.join_default);
                line.push_str(if tree.children.is_empty() {
                    fmt.padding.join_default
                } else {
                    fmt.padding.join_haschild
                });
                line.push_str(fmt.padding.join_default);
                line.push_str(fmt.padding.join_default);
            }
            line.push_str(&fmt.print_tag(tree.tag, &tree.name));
            let protector = protected_tags.get(&tree.tag);
            line.push_str(fmt.print_protector(protector));
            // Push the line to the accumulator then recurse.
            acc.push(line);
            let nb_children = tree.children.len();
            for (i, child) in tree.children.iter().enumerate() {
                indent.increment(fmt, is_last_child);
                print_aux(child, padding, fmt, indent, protected_tags, i + 1 == nb_children, acc);
                indent.decrement(fmt);
            }
        }
    }
}

const DEFAULT_FORMATTER: DisplayFmt = DisplayFmt {
    wrapper: DisplayFmtWrapper {
        top: '─',
        bot: '─',
        warning_text: "Warning: this tree is indicative only. Some tags may have been hidden.",
    },
    perm: DisplayFmtPermission { open: "|", sep: "|", close: "|", uninit: "---", range_sep: ".." },
    padding: DisplayFmtPadding {
        join_middle: "├",
        join_last: "└",
        indent_middle: "│ ",
        indent_last: "  ",
        join_haschild: "┬",
        join_default: "─",
    },
    accessed: DisplayFmtAccess { yes: " ", no: "?", meh: "-" },
};

impl<'tcx> Tree {
    /// Display the contents of the tree.
    pub fn print_tree(
        &self,
        protected_tags: &FxHashMap<BorTag, ProtectorKind>,
        show_unnamed: bool,
    ) -> InterpResult<'tcx> {
        let mut indenter = DisplayIndent::new();
        let ranges = self.rperms.iter_all().map(|(range, _perms)| range).collect::<Vec<_>>();
        if let Some(repr) = DisplayRepr::from(self, show_unnamed) {
            repr.print(
                &DEFAULT_FORMATTER,
                &mut indenter,
                protected_tags,
                ranges,
                /* print warning message about tags not shown */ !show_unnamed,
            );
        }
        Ok(())
    }
}
