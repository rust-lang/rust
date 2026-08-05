//! Implementation of applying changes to a syntax tree.

use std::{cmp::Ordering, ops::Range};

use rowan::TextRange;
use rustc_hash::FxHashMap;
use stdx::format_to;

use crate::{NodeOrToken, SyntaxElement, SyntaxNode};

use super::{
    Change, ChangeKind, PositionRepr, SyntaxAnnotation, SyntaxEdit, SyntaxEditor, SyntaxMapping,
    mapping::MissingMapping,
};

/// A validated batch of changes in the exact order in which it must execute.
///
/// Planning is deliberately separate from tree mutation. Once an `EditPlan`
/// exists, execution does not need to reason about overlaps, dependencies, or
/// source ordering.
struct EditPlan {
    changes: Vec<PlannedChange>,
}

/// A change whose target tree and output tracking are fully known.
struct PlannedChange {
    tree: SyntaxNode,
    change: Change,
    record_as_changed: bool,
}

impl PlannedChange {
    /// Returns the immutable source elements that this change will slice in.
    fn replacement_elements(&self) -> &[SyntaxElement] {
        match &self.change {
            Change::Insert(_, element) | Change::Replace(_, Some(element)) => {
                std::slice::from_ref(element)
            }
            Change::InsertAll(_, elements)
            | Change::ReplaceWithMany(_, elements)
            | Change::ReplaceAll(_, elements) => elements,
            Change::Replace(_, None) => &[],
        }
    }
}

/// The dependency info accumulated from one source ordered changes.
///
/// `parent` is an edge to the nearest containing node replacement. A discarded
/// entry has no executable graph node because an ancestor deletion or ambiguous
/// range replacement has mde its target unavailable.
#[derive(Clone, Copy, Default)]
struct PlanEntry {
    parent: Option<usize>,
    discarded: bool,
}

/// Planning failure containing the source-ordered changes use for diag.
struct InvalidEditPlan {
    changes: Vec<Change>,
}

impl EditPlan {
    /// Validates raw editor changes and turns them into an execution schedule.
    ///
    /// The input is first sorted in source order, dependent targets are then
    /// rewritten from their input trees into ancestor replacement trees. Finally,
    /// discarded changes are removed and the dependency forest is traversed in
    /// postorder.
    /// Independent roots and sibling changes are prioritized right to left.
    fn build(
        mut changes: Vec<Change>,
        mappings: &SyntaxMapping,
        mut node_depth: impl FnMut(SyntaxNode) -> usize,
    ) -> Result<Self, InvalidEditPlan> {
        changes.sort_by(|left, right| {
            left.target_range()
                .start()
                .cmp(&right.target_range().start())
                .then_with(|| {
                    let left_target = left.target_parent();
                    let right_target = right.target_parent();
                    if left_target == right_target {
                        Ordering::Equal
                    } else {
                        node_depth(left_target).cmp(&node_depth(right_target))
                    }
                })
                .then(left.change_kind().cmp(&right.change_kind()))
        });

        if !Self::replacements_are_disjoint(&changes, &mut node_depth) {
            return Err(InvalidEditPlan { changes });
        }

        let mut entries = vec![PlanEntry::default(); changes.len()];
        let mut regions_by_tree = FxHashMap::<SyntaxNode, Vec<ChangedRegion>>::default();

        for (index, change) in changes.iter().enumerate() {
            let target_tree = change.target_parent().tree_top();
            let regions = regions_by_tree.entry(target_tree).or_default();
            if let Some(region_index) = regions
                .iter()
                .rposition(|region| region.range.contains_range(change.target_range()))
            {
                regions.truncate(region_index + 1);
                match regions[region_index].nested_changes {
                    NestedChanges::Remap => {
                        entries[index].parent = Some(regions[region_index].change_index);
                    }
                    NestedChanges::Discard => entries[index].discarded = true,
                }
            } else {
                regions.clear();
            }

            if let Some(region) = ChangedRegion::for_change(change, index, entries[index].discarded)
            {
                regions.push(region);
            }
        }

        // Work from the innermost dependency towards the outermost one. This
        // lets a chain A -> B -> C rewrite C into B before B itself is mapped
        // into A's replacement tree.
        for (child, entry) in entries.iter().enumerate().rev() {
            if let Some(parent) = entry.parent {
                Self::rewrite_dependent_target(&mut changes, parent, child, mappings);
            }
        }

        let mut children = vec![Vec::new(); changes.len()];
        for (child, entry) in entries.iter().enumerate() {
            if let Some(parent) = entry.parent {
                children[parent].push(child);
            }
        }
        for siblings in &mut children {
            siblings.sort_by(|&left, &right| {
                Self::execution_priority(&changes[left], &changes[right], &mut node_depth)
            });
        }

        let mut roots = entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                (!entry.discarded && entry.parent.is_none()).then_some(index)
            })
            .collect::<Vec<_>>();
        roots.sort_by(|&left, &right| {
            Self::execution_priority(&changes[left], &changes[right], &mut node_depth)
        });

        let mut planned = changes
            .into_iter()
            .zip(entries)
            .map(|(change, entry)| {
                (!entry.discarded).then_some(PlannedChange {
                    tree: change.target_parent().tree_top(),
                    change,
                    record_as_changed: entry.parent.is_none(),
                })
            })
            .collect::<Vec<_>>();

        let mut ordered = Vec::new();
        for root in roots {
            Self::append_postorder(root, &children, &mut planned, &mut ordered);
        }

        Ok(Self { changes: ordered })
    }

    /// Orders disjoint changes from right to left, with deeper ties first.
    fn execution_priority(
        left: &Change,
        right: &Change,
        node_depth: &mut impl FnMut(SyntaxNode) -> usize,
    ) -> Ordering {
        right
            .target_range()
            .start()
            .cmp(&left.target_range().start())
            .then_with(|| node_depth(right.target_parent()).cmp(&node_depth(left.target_parent())))
            .then(right.change_kind().cmp(&left.change_kind()))
    }

    /// Appends a dependency subtree in post order fashion.
    fn append_postorder(
        index: usize,
        children: &[Vec<usize>],
        planned: &mut [Option<PlannedChange>],
        ordered: &mut Vec<PlannedChange>,
    ) {
        for &child in &children[index] {
            Self::append_postorder(child, children, planned, ordered);
        }
        ordered.push(planned[index].take().expect("reachable plan nodes are not discarded"));
    }

    /// Checks that replacement at the same tree depth do not overlap
    ///
    /// `changes` is sorted by range start, so overlap is a single comparison against the
    /// last range at that key, and `insert` can throw away the range it evicts.
    fn replacements_are_disjoint(
        changes: &[Change],
        mut node_depth: impl FnMut(SyntaxNode) -> usize,
    ) -> bool {
        let mut previous = FxHashMap::<(SyntaxNode, usize), TextRange>::default();
        for change in changes {
            if !matches!(change.change_kind(), ChangeKind::Replace | ChangeKind::ReplaceRange) {
                continue;
            }

            let parent = change.target_parent();
            let key = (parent.tree_top(), node_depth(parent));
            if previous
                .insert(key, change.target_range())
                .is_some_and(|range| range.end() > change.target_range().start())
            {
                return false;
            }
        }
        true
    }

    /// Maps one dependent change into its ancestor replacement tree.
    fn rewrite_dependent_target(
        changes: &mut [Change],
        parent: usize,
        child: usize,
        mappings: &SyntaxMapping,
    ) {
        let (input_ancestor, output_ancestor) = match &changes[parent] {
            Change::Replace(
                SyntaxElement::Node(target),
                Some(SyntaxElement::Node(replacement)),
            ) => (target.clone(), replacement.clone()),
            _ => unreachable!("only node replacements can own dependent changes"),
        };

        let upmap_node = |target: &SyntaxNode| {
            mappings.upmap_child(target, &input_ancestor, &output_ancestor).unwrap_or_else(
                |MissingMapping(current)| {
                    panic!(
                        "no mappings exist between {current:?} (ancestor of {input_ancestor:?}) and {output_ancestor:?}"
                    )
                },
            )
        };
        let upmap_element = |target: &SyntaxElement| {
            mappings.upmap_child_element(target, &input_ancestor, &output_ancestor).unwrap_or_else(
                |MissingMapping(current)| {
                    panic!(
                        "no mappings exist between {current:?} (ancestor of {input_ancestor:?}) and {output_ancestor:?}"
                    )
                },
            )
        };

        match &mut changes[child] {
            Change::Insert(position, _) | Change::InsertAll(position, _) => {
                match &mut position.repr {
                    PositionRepr::FirstChild(parent) => *parent = upmap_node(parent),
                    PositionRepr::After(child) => *child = upmap_element(child),
                }
            }
            Change::Replace(target, _) | Change::ReplaceWithMany(target, _) => {
                *target = upmap_element(target);
            }
            Change::ReplaceAll(range, _) => {
                *range = upmap_element(range.start())..=upmap_element(range.end());
            }
        }
    }
}

/// A stable structural address expressed as `children_with_token` indices.
#[derive(Clone)]
struct SyntaxPath {
    child_indices: Vec<usize>,
}

impl SyntaxPath {
    /// Builds the root-relative path of element in its current tree.
    fn new(element: &SyntaxElement) -> Self {
        let mut child_indices = Vec::new();
        let mut node = match element {
            SyntaxElement::Node(node) => node.clone(),
            SyntaxElement::Token(token) => {
                child_indices.push(token.index());
                token.parent().unwrap()
            }
        };

        while let Some(parent) = node.parent() {
            child_indices.push(node.index());
            node = parent;
        }
        child_indices.reverse();
        Self { child_indices }
    }

    /// Follows this path from root, returning None if the structure differs.
    fn resolve(&self, root: &SyntaxNode) -> Option<SyntaxElement> {
        let mut current = SyntaxElement::Node(root.clone());
        for &index in &self.child_indices {
            current = current.into_node()?.children_with_tokens().nth(index)?;
        }
        Some(current)
    }

    /// Removes `ancestor`'s prefix, yielding this path within that subtree.
    ///
    /// Could have used LCA?
    fn relative_to(&self, ancestor: &SyntaxPath) -> Option<SyntaxPath> {
        self.child_indices
            .strip_prefix(ancestor.child_indices.as_slice())
            .map(|relative| SyntaxPath { child_indices: relative.to_vec() })
    }

    /// Appends an inserted child slot and a path relative to that child.
    fn in_child(&self, index: usize, relative: &SyntaxPath) -> SyntaxPath {
        let mut child_indices =
            Vec::with_capacity(self.child_indices.len() + relative.child_indices.len() + 1);
        child_indices.extend_from_slice(&self.child_indices);
        child_indices.push(index);
        child_indices.extend_from_slice(&relative.child_indices);
        SyntaxPath { child_indices }
    }

    /// Updates this path for a splice and reports whether its element survives.
    fn adjust_for_splice(
        &mut self,
        parent: &SyntaxPath,
        deleted: &Range<usize>,
        inserted: usize,
    ) -> bool {
        let Some(relative) = self.child_indices.strip_prefix(parent.child_indices.as_slice())
        else {
            return true;
        };
        let Some((&child, _)) = relative.split_first() else { return true };

        if deleted.contains(&child) {
            return false;
        }
        if child >= deleted.end {
            let new_child = child + inserted;
            self.child_indices[parent.child_indices.len()] =
                new_child - (deleted.end - deleted.start);
        }
        true
    }
}

/// An annotation paired with its structural location and registration order.
#[derive(Clone)]
struct TrackedAnnotation {
    path: SyntaxPath,
    annotation: SyntaxAnnotation,
    order: usize,
}

/// A structural edit used to translate original paths into a current tree.
enum PathEdit {
    /// A child-list splice with all coordinates relative to the pre-edit tree.
    Splice { parent: SyntaxPath, deleted: Range<usize>, inserted: usize },
    /// A root replacement, after which no path into the old root survives.
    ReplaceRoot,
}

/// The evolving immutable root and location metadata for one source tree.
///
/// A syntax edit can involve the editor root plus several detached factory
/// trees. Each receives an independent state so dependent edits can be applied
/// before a generated tree is inserted elsewhere.
struct TreeState {
    root: SyntaxNode,
    edits: Vec<PathEdit>,
    changed: Vec<SyntaxPath>,
    original_annotations: Vec<TrackedAnnotation>,
    annotations: Vec<TrackedAnnotation>,
}

impl TreeState {
    /// Starts tracking an unmodified immutable root.
    fn new(root: SyntaxNode) -> Self {
        Self {
            root,
            edits: Vec::new(),
            changed: Vec::new(),
            original_annotations: Vec::new(),
            annotations: Vec::new(),
        }
    }

    /// Replay structural edits to translate an original path into this state.
    fn map_original_path(&self, mut path: SyntaxPath) -> Option<SyntaxPath> {
        for edit in &self.edits {
            match edit {
                PathEdit::Splice { parent, deleted, inserted } => {
                    if !path.adjust_for_splice(parent, deleted, *inserted) {
                        return None;
                    }
                }
                PathEdit::ReplaceRoot => return None,
            }
        }
        Some(path)
    }

    /// Finds a change target in the current root.
    fn map_original_element(&self, element: &SyntaxElement) -> SyntaxElement {
        self.map_original_path(SyntaxPath::new(element))
            .and_then(|path| path.resolve(&self.root))
            .expect("an edit target must still be present")
    }

    /// Applies one child-list splice and updates tracked structural path.
    fn splice(
        &mut self,
        parent_path: SyntaxPath,
        deleted: Range<usize>,
        inserted: Vec<PreparedElement>,
        track_as_changed: bool,
    ) {
        let inserted_count = inserted.len();
        self.changed
            .retain_mut(|path| path.adjust_for_splice(&parent_path, &deleted, inserted_count));
        self.annotations
            .retain_mut(|it| it.path.adjust_for_splice(&parent_path, &deleted, inserted_count));

        for (offset, element) in inserted.iter().enumerate() {
            let index = deleted.start + offset;
            if track_as_changed {
                self.changed
                    .push(parent_path.in_child(index, &SyntaxPath { child_indices: Vec::new() }));
            }
            self.annotations.extend(element.annotations.iter().map(|annotation| {
                TrackedAnnotation {
                    path: parent_path.in_child(index, &annotation.path),
                    annotation: annotation.annotation,
                    order: annotation.order,
                }
            }));
        }

        let parent = parent_path.resolve(&self.root).and_then(SyntaxElement::into_node).unwrap();
        let green = rowan::GreenNodeData::splice_children(
            parent.green(),
            deleted.clone(),
            inserted.into_iter().map(PreparedElement::into_green),
        );
        self.root = SyntaxNode::new_root(parent.replace_with(green));
        self.edits.push(PathEdit::Splice {
            parent: parent_path,
            deleted,
            inserted: inserted_count,
        });
    }

    /// Replaces the tree's root with a prepared node payload.
    fn replace_root(&mut self, replacement: PreparedElement, track_as_changed: bool) {
        let NodeOrToken::Node(node) = replacement.syntax else {
            panic!("root node replacement should be a node")
        };
        self.root = SyntaxNode::new_root(node.green().to_owned());
        self.changed.clear();
        if track_as_changed {
            self.changed.push(SyntaxPath { child_indices: Vec::new() });
        }
        self.annotations = replacement.annotations;
        self.edits.push(PathEdit::ReplaceRoot);
    }

    /// Applies a planned change to this tree using already prepared payloads.
    fn apply(
        &mut self,
        change: &Change,
        replacement: Vec<PreparedElement>,
        record_as_changed: bool,
    ) {
        match change {
            Change::Insert(position, _) | Change::InsertAll(position, _) => {
                let (parent, index) = match &position.repr {
                    PositionRepr::FirstChild(parent) => {
                        let parent = self.map_original_element(&parent.clone().into());
                        (parent.into_node().unwrap(), 0)
                    }
                    PositionRepr::After(child) => {
                        let child = self.map_original_element(child);
                        (child.parent().unwrap(), child.index() + 1)
                    }
                };
                self.splice(
                    SyntaxPath::new(&parent.into()),
                    index..index,
                    replacement,
                    record_as_changed,
                );
            }
            Change::Replace(SyntaxElement::Node(target), Some(_)) if target.parent().is_none() => {
                self.replace_root(replacement.into_iter().next().unwrap(), record_as_changed);
            }
            Change::Replace(target, _) | Change::ReplaceWithMany(target, _) => {
                let target = self.map_original_element(target);
                let parent = target.parent().unwrap();
                let index = target.index();
                self.splice(
                    SyntaxPath::new(&parent.into()),
                    index..index + 1,
                    replacement,
                    record_as_changed,
                );
            }
            Change::ReplaceAll(range, _) => {
                let start = self.map_original_element(range.start());
                let end = self.map_original_element(range.end());
                let parent = start.parent().unwrap();
                self.splice(
                    SyntaxPath::new(&parent.into()),
                    start.index()..end.index() + 1,
                    replacement,
                    record_as_changed,
                );
            }
        }
    }
}

/// A replacement payload paired with the annotation below it.
///
/// The syntax element remains an immutable snapshot of its source tree.
/// Annotation paths are relative to the payload root and are rebased by
/// splice
struct PreparedElement {
    syntax: SyntaxElement,
    annotations: Vec<TrackedAnnotation>,
}

impl PreparedElement {
    fn into_green(self) -> rowan::NodeOrToken<rowan::GreenNode, rowan::GreenToken> {
        match self.syntax {
            SyntaxElement::Node(node) => NodeOrToken::Node(node.green().to_owned()),
            SyntaxElement::Token(token) => NodeOrToken::Token(token.green().to_owned()),
        }
    }
}

/// Owns all evolving trees involved in executing an edit plan.
struct TreeStore {
    states: FxHashMap<SyntaxNode, TreeState>,
}

impl TreeStore {
    /// Creates per tree state for annotations after following factory mapping.
    fn with_annotations(
        annotations: Vec<(SyntaxElement, SyntaxAnnotation)>,
        mappings: &SyntaxMapping,
    ) -> Self {
        let mut states = FxHashMap::<SyntaxNode, TreeState>::default();
        for (order, (element, annotation)) in annotations.into_iter().enumerate() {
            let element = mappings.upmap_element(&element);
            let tree = element.tree_top();
            let tracked = TrackedAnnotation { path: SyntaxPath::new(&element), annotation, order };
            let state = states.entry(tree.clone()).or_insert_with(|| TreeState::new(tree));
            state.original_annotations.push(tracked.clone());
            state.annotations.push(tracked);
        }
        Self { states }
    }

    /// Execute an already ordered plan without performing further analysis.
    fn execute(&mut self, plan: EditPlan) {
        for planned in plan.changes {
            self.states
                .entry(planned.tree.clone())
                .or_insert_with(|| TreeState::new(planned.tree.clone()));
            let replacement = planned
                .replacement_elements()
                .iter()
                .map(|element| self.prepare_element(element))
                .collect();
            self.states.get_mut(&planned.tree).unwrap().apply(
                &planned.change,
                replacement,
                planned.record_as_changed,
            );
        }
    }

    /// Captures the source element and annotations used by a replacement.
    fn prepare_element(&self, element: &SyntaxElement) -> PreparedElement {
        let tree = element.tree_top();
        let original_path = SyntaxPath::new(element);
        let (element, annotations) = match self.states.get(&tree) {
            Some(state) => {
                let annotations_below =
                    |annotations: &[TrackedAnnotation], ancestor: &SyntaxPath| {
                        annotations
                            .iter()
                            .filter_map(|annotation| {
                                annotation.path.relative_to(ancestor).map(|path| {
                                    TrackedAnnotation {
                                        path,
                                        annotation: annotation.annotation,
                                        order: annotation.order,
                                    }
                                })
                            })
                            .collect()
                    };
                match state.map_original_path(original_path.clone()) {
                    Some(path) => {
                        let element = path.resolve(&state.root).unwrap();
                        let annotations = annotations_below(&state.annotations, &path);
                        (element, annotations)
                    }
                    None => {
                        let annotations =
                            annotations_below(&state.original_annotations, &original_path);
                        (element.clone(), annotations)
                    }
                }
            }
            None => (element.clone(), Vec::new()),
        };
        PreparedElement { syntax: element, annotations }
    }

    /// Resolves the editor roots tracked paths and constructs the public edit.
    fn finish(mut self, old_root: SyntaxNode) -> SyntaxEdit {
        let state =
            self.states.remove(&old_root).unwrap_or_else(|| TreeState::new(old_root.clone()));
        let new_root = state.root;

        let mut changed_elements = state
            .changed
            .into_iter()
            .filter_map(|path| path.resolve(&new_root))
            .collect::<Vec<_>>();
        changed_elements.sort_by_key(|element| element.text_range().start());

        let mut annotations = FxHashMap::<SyntaxAnnotation, Vec<(usize, SyntaxElement)>>::default();
        for annotation in state.annotations {
            if let Some(element) = annotation.path.resolve(&new_root) {
                annotations
                    .entry(annotation.annotation)
                    .or_default()
                    .push((annotation.order, element));
            }
        }
        let annotations = annotations
            .into_iter()
            .map(|(annotation, mut elements)| {
                elements.sort_by_key(|(order, element)| (*order, element.text_range().start()));
                (annotation, elements.into_iter().map(|(_, element)| element).collect())
            })
            .collect();

        SyntaxEdit { old_root, new_root, changed_elements, annotations }
    }
}

/// Plans and executes all changes recorded by a SyntaxEditor.
pub(super) fn apply_edits(editor: SyntaxEditor) -> SyntaxEdit {
    let SyntaxEditor { root, changes, annotations, make } = editor;
    let mappings = make.take();
    let mut node_depths = FxHashMap::<SyntaxNode, usize>::default();
    let mut node_depth = |node: SyntaxNode| {
        *node_depths.entry(node).or_insert_with_key(|node| node.ancestors().count())
    };

    let plan = match EditPlan::build(changes.into_inner(), &mappings, &mut node_depth) {
        Ok(plan) => plan,
        Err(InvalidEditPlan { changes }) => {
            report_intersecting_changes(&changes, &mut node_depth, &root);
            return SyntaxEdit {
                old_root: root.clone(),
                new_root: root,
                annotations: FxHashMap::default(),
                changed_elements: Vec::new(),
            };
        }
    };

    let mut trees = TreeStore::with_annotations(annotations.into_inner(), &mappings);
    trees.execute(plan);
    trees.finish(root)
}

fn report_intersecting_changes(
    changes: &[Change],
    mut get_node_depth: impl FnMut(SyntaxNode) -> usize,
    root: &SyntaxNode,
) {
    let intersecting_changes = changes
        .iter()
        .zip(changes.iter().skip(1))
        .filter(|(l, r)| {
            // We only care about checking for disjoint replace ranges.
            matches!(
                (l.change_kind(), r.change_kind()),
                (
                    ChangeKind::Replace | ChangeKind::ReplaceRange,
                    ChangeKind::Replace | ChangeKind::ReplaceRange
                )
            )
        })
        .filter(|(l, r)| {
            get_node_depth(l.target_parent()) == get_node_depth(r.target_parent())
                && (l.target_range().end() > r.target_range().start())
        });

    let mut error_msg = String::from("some replace change ranges intersect!\n");

    let parent_str = root.to_string();

    for (l, r) in intersecting_changes {
        let mut highlighted_str = parent_str.clone();
        let l_range = l.target_range();
        let r_range = r.target_range();

        let i_range = l_range.intersect(r_range).unwrap();
        let i_str = format!("\x1b[46m{}", &parent_str[i_range]);

        let pre_range: Range<usize> = l_range.start().into()..i_range.start().into();
        let pre_str = format!("\x1b[44m{}", &parent_str[pre_range]);

        let (highlight_range, highlight_str) = if l_range == r_range {
            format_to!(error_msg, "\x1b[46mleft change:\x1b[0m  {l:?} {l}\n");
            format_to!(error_msg, "\x1b[46mequals\x1b[0m\n");
            format_to!(error_msg, "\x1b[46mright change:\x1b[0m {r:?} {r}\n");
            let i_highlighted = format!("{i_str}\x1b[0m\x1b[K");
            let total_range: Range<usize> = i_range.into();
            (total_range, i_highlighted)
        } else {
            format_to!(error_msg, "\x1b[44mleft change:\x1b[0m  {l:?} {l}\n");
            let range_end = if l_range.contains_range(r_range) {
                format_to!(error_msg, "\x1b[46mcovers\x1b[0m\n");
                format_to!(error_msg, "\x1b[46mright change:\x1b[0m {r:?} {r}\n");
                l_range.end()
            } else {
                format_to!(error_msg, "\x1b[46mintersects\x1b[0m\n");
                format_to!(error_msg, "\x1b[42mright change:\x1b[0m {r:?} {r}\n");
                r_range.end()
            };

            let post_range: Range<usize> = i_range.end().into()..range_end.into();

            let post_str = format!("\x1b[42m{}", &parent_str[post_range]);
            let result = format!("{pre_str}{i_str}{post_str}\x1b[0m\x1b[K");
            let total_range: Range<usize> = l_range.start().into()..range_end.into();
            (total_range, result)
        };
        highlighted_str.replace_range(highlight_range, &highlight_str);

        format_to!(error_msg, "{highlighted_str}\n");
    }

    stdx::always!(false, "{}", error_msg);
}

/// A replacement region that can contain later source ordered changeds
struct ChangedRegion {
    range: TextRange,
    change_index: usize,
    nested_changes: NestedChanges,
}

/// How changes nested within a replacement region are handled.
enum NestedChanges {
    /// Map nested targets into a one-to-one node replacement.
    Remap,
    /// Drop nested changes because the replacement has no unique counterpart.
    Discard,
}

impl ChangedRegion {
    /// Describes a region replaced by change, if it can contain changes.
    fn for_change(change: &Change, change_index: usize, discarded: bool) -> Option<Self> {
        match change {
            Change::Replace(SyntaxElement::Node(target), replacement) => Some(Self {
                range: target.text_range(),
                change_index,
                nested_changes: if !discarded && matches!(replacement, Some(SyntaxElement::Node(_)))
                {
                    NestedChanges::Remap
                } else {
                    NestedChanges::Discard
                },
            }),
            Change::ReplaceWithMany(SyntaxElement::Node(target), _) => Some(Self {
                range: target.text_range(),
                change_index,
                nested_changes: NestedChanges::Discard,
            }),
            Change::ReplaceAll(elements, _) => Some(Self {
                range: TextRange::new(
                    elements.start().text_range().start(),
                    elements.end().text_range().end(),
                ),
                change_index,
                nested_changes: NestedChanges::Discard,
            }),
            _ => None,
        }
    }
}
