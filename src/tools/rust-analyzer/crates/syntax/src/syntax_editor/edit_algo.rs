//! Implementation of applying changes to a syntax tree.

use std::{cmp::Ordering, collections::VecDeque, ops::RangeInclusive};

use rowan::TextRange;
use rustc_hash::FxHashMap;

use crate::{
    syntax_editor::{mapping::MissingMapping, Change, ChangeKind, PositionRepr},
    SyntaxElement, SyntaxNode, SyntaxNodePtr,
};

use super::{SyntaxEdit, SyntaxEditor};

pub(super) fn apply_edits(editor: SyntaxEditor) -> SyntaxEdit {
    // Algorithm overview:
    //
    // - Sort changes by (range, type)
    //   - Ensures that parent edits are before child edits
    //   - Ensures that inserts will be guaranteed to be inserted at the right range
    // - Validate changes
    //   - Checking for invalid changes is easy since the changes will be sorted by range
    // - Fixup change targets
    //   - standalone change? map to original syntax tree
    //   - dependent change?
    //     - try to map to parent change (either independent or another dependent)
    //     - note: need to keep track of a parent change stack, since a change can be a parent of multiple changes
    // - Apply changes
    //   - find changes to apply to real tree by applying nested changes first
    //   - changed nodes become part of the changed node set (useful for the formatter to only change those parts)
    // - Propagate annotations

    let SyntaxEditor { root, mut changes, mappings, annotations } = editor;

    let mut node_depths = FxHashMap::<SyntaxNode, usize>::default();
    let mut get_node_depth = |node: SyntaxNode| {
        *node_depths.entry(node).or_insert_with_key(|node| node.ancestors().count())
    };

    // Sort changes by range, then depth, then change kind, so that we can:
    // - ensure that parent edits are ordered before child edits
    // - ensure that inserts will be guaranteed to be inserted at the right range
    // - easily check for disjoint replace ranges
    changes.sort_by(|a, b| {
        a.target_range()
            .start()
            .cmp(&b.target_range().start())
            .then_with(|| {
                let a_target = a.target_parent();
                let b_target = b.target_parent();

                if a_target == b_target {
                    return Ordering::Equal;
                }

                get_node_depth(a_target).cmp(&get_node_depth(b_target))
            })
            .then(a.change_kind().cmp(&b.change_kind()))
    });

    let disjoint_replaces_ranges = changes
        .iter()
        .zip(changes.iter().skip(1))
        .filter(|(l, r)| {
            // We only care about checking for disjoint replace ranges
            matches!(
                (l.change_kind(), r.change_kind()),
                (
                    ChangeKind::Replace | ChangeKind::ReplaceRange,
                    ChangeKind::Replace | ChangeKind::ReplaceRange
                )
            )
        })
        .all(|(l, r)| {
            get_node_depth(l.target_parent()) != get_node_depth(r.target_parent())
                || l.target_range().intersect(r.target_range()).is_none()
        });

    if stdx::never!(
        !disjoint_replaces_ranges,
        "some replace change ranges intersect: {:?}",
        changes
    ) {
        return SyntaxEdit {
            old_root: root.clone(),
            new_root: root,
            annotations: Default::default(),
            changed_elements: vec![],
        };
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct DependentChange {
        parent: u32,
        child: u32,
    }

    // Build change tree
    let mut changed_ancestors: VecDeque<ChangedAncestor> = VecDeque::new();
    let mut dependent_changes = vec![];
    let mut independent_changes = vec![];

    for (change_index, change) in changes.iter().enumerate() {
        // Check if this change is dependent on another change (i.e. it's contained within another range)
        if let Some(index) = changed_ancestors
            .iter()
            .rev()
            .position(|ancestor| ancestor.affected_range().contains_range(change.target_range()))
        {
            // Pop off any ancestors that aren't applicable
            changed_ancestors.drain((index + 1)..);

            // FIXME: Resolve changes that depend on a range of elements
            let ancestor = &changed_ancestors[index];

            dependent_changes.push(DependentChange {
                parent: ancestor.change_index as u32,
                child: change_index as u32,
            });
        } else {
            // This change is independent of any other change

            // Drain the changed ancestors since we're no longer in a set of dependent changes
            changed_ancestors.drain(..);

            independent_changes.push(change_index as u32);
        }

        // Add to changed ancestors, if applicable
        match change {
            Change::Insert(_, _) | Change::InsertAll(_, _) => {}
            Change::Replace(target, _) | Change::ReplaceWithMany(target, _) => {
                changed_ancestors.push_back(ChangedAncestor::single(target, change_index))
            }
            Change::ReplaceAll(range, _) => {
                changed_ancestors.push_back(ChangedAncestor::multiple(range, change_index))
            }
        }
    }

    // Map change targets to the correct syntax nodes
    let tree_mutator = TreeMutator::new(&root);
    let mut changed_elements = vec![];

    for index in independent_changes {
        match &mut changes[index as usize] {
            Change::Insert(target, _) | Change::InsertAll(target, _) => {
                match &mut target.repr {
                    PositionRepr::FirstChild(parent) => {
                        *parent = tree_mutator.make_syntax_mut(parent);
                    }
                    PositionRepr::After(child) => {
                        *child = tree_mutator.make_element_mut(child);
                    }
                };
            }
            Change::Replace(target, _) | Change::ReplaceWithMany(target, _) => {
                *target = tree_mutator.make_element_mut(target);
            }
            Change::ReplaceAll(range, _) => {
                let start = tree_mutator.make_element_mut(range.start());
                let end = tree_mutator.make_element_mut(range.end());

                *range = start..=end;
            }
        }

        // Collect changed elements
        match &changes[index as usize] {
            Change::Insert(_, element) => changed_elements.push(element.clone()),
            Change::InsertAll(_, elements) => changed_elements.extend(elements.iter().cloned()),
            Change::Replace(_, Some(element)) => changed_elements.push(element.clone()),
            Change::Replace(_, None) => {}
            Change::ReplaceWithMany(_, elements) => {
                changed_elements.extend(elements.iter().cloned())
            }
            Change::ReplaceAll(_, elements) => changed_elements.extend(elements.iter().cloned()),
        }
    }

    for DependentChange { parent, child } in dependent_changes.into_iter() {
        let (input_ancestor, output_ancestor) = match &changes[parent as usize] {
            // No change will depend on an insert since changes can only depend on nodes in the root tree
            Change::Insert(_, _) | Change::InsertAll(_, _) => unreachable!(),
            Change::Replace(target, Some(new_target)) => {
                (to_owning_node(target), to_owning_node(new_target))
            }
            // Silently drop outdated change
            Change::Replace(_, None) => continue,
            Change::ReplaceAll(_, _) | Change::ReplaceWithMany(_, _) => {
                unimplemented!("cannot resolve changes that depend on replacing many elements")
            }
        };

        let upmap_target_node = |target: &SyntaxNode| {
            match mappings.upmap_child(target, &input_ancestor, &output_ancestor) {
                Ok(it) => it,
                Err(MissingMapping(current)) => unreachable!("no mappings exist between {current:?} (ancestor of {input_ancestor:?}) and {output_ancestor:?}"),
            }
        };

        let upmap_target = |target: &SyntaxElement| {
            match mappings.upmap_child_element(target, &input_ancestor, &output_ancestor) {
                Ok(it) => it,
                Err(MissingMapping(current)) => unreachable!("no mappings exist between {current:?} (ancestor of {input_ancestor:?}) and {output_ancestor:?}"),
            }
        };

        match &mut changes[child as usize] {
            Change::Insert(target, _) | Change::InsertAll(target, _) => match &mut target.repr {
                PositionRepr::FirstChild(parent) => {
                    *parent = upmap_target_node(parent);
                }
                PositionRepr::After(child) => {
                    *child = upmap_target(child);
                }
            },
            Change::Replace(target, _) | Change::ReplaceWithMany(target, _) => {
                *target = upmap_target(target);
            }
            Change::ReplaceAll(range, _) => {
                *range = upmap_target(range.start())..=upmap_target(range.end());
            }
        }
    }

    // Apply changes
    let mut root = tree_mutator.mutable_clone;

    for change in changes {
        match change {
            Change::Insert(position, element) => {
                let (parent, index) = position.place();
                parent.splice_children(index..index, vec![element]);
            }
            Change::InsertAll(position, elements) => {
                let (parent, index) = position.place();
                parent.splice_children(index..index, elements);
            }
            Change::Replace(target, None) => {
                target.detach();
            }
            Change::Replace(SyntaxElement::Node(target), Some(new_target)) if target == root => {
                root = new_target.into_node().expect("root node replacement should be a node");
            }
            Change::Replace(target, Some(new_target)) => {
                let parent = target.parent().unwrap();
                parent.splice_children(target.index()..target.index() + 1, vec![new_target]);
            }
            Change::ReplaceWithMany(target, elements) => {
                let parent = target.parent().unwrap();
                parent.splice_children(target.index()..target.index() + 1, elements);
            }
            Change::ReplaceAll(range, elements) => {
                let start = range.start().index();
                let end = range.end().index();
                let parent = range.start().parent().unwrap();
                parent.splice_children(start..end + 1, elements);
            }
        }
    }

    // Propagate annotations
    let annotations = annotations.into_iter().filter_map(|(element, annotation)| {
        match mappings.upmap_element(&element, &root) {
            // Needed to follow the new tree to find the resulting element
            Some(Ok(mapped)) => Some((mapped, annotation)),
            // Element did not need to be mapped
            None => Some((element, annotation)),
            // Element did not make it to the final tree
            Some(Err(_)) => None,
        }
    });

    let mut annotation_groups = FxHashMap::default();

    for (element, annotation) in annotations {
        annotation_groups.entry(annotation).or_insert(vec![]).push(element);
    }

    SyntaxEdit {
        old_root: tree_mutator.immutable,
        new_root: root,
        changed_elements,
        annotations: annotation_groups,
    }
}

fn to_owning_node(element: &SyntaxElement) -> SyntaxNode {
    match element {
        SyntaxElement::Node(node) => node.clone(),
        SyntaxElement::Token(token) => token.parent().unwrap().clone(),
    }
}

struct ChangedAncestor {
    kind: ChangedAncestorKind,
    change_index: usize,
}

enum ChangedAncestorKind {
    Single { node: SyntaxNode },
    Range { _changed_elements: RangeInclusive<SyntaxElement>, _in_parent: SyntaxNode },
}

impl ChangedAncestor {
    fn single(element: &SyntaxElement, change_index: usize) -> Self {
        let kind = match element {
            SyntaxElement::Node(node) => ChangedAncestorKind::Single { node: node.clone() },
            SyntaxElement::Token(token) => {
                ChangedAncestorKind::Single { node: token.parent().unwrap() }
            }
        };

        Self { kind, change_index }
    }

    fn multiple(range: &RangeInclusive<SyntaxElement>, change_index: usize) -> Self {
        Self {
            kind: ChangedAncestorKind::Range {
                _changed_elements: range.clone(),
                _in_parent: range.start().parent().unwrap(),
            },
            change_index,
        }
    }

    fn affected_range(&self) -> TextRange {
        match &self.kind {
            ChangedAncestorKind::Single { node } => node.text_range(),
            ChangedAncestorKind::Range { _changed_elements: changed_nodes, _in_parent: _ } => {
                TextRange::new(
                    changed_nodes.start().text_range().start(),
                    changed_nodes.end().text_range().end(),
                )
            }
        }
    }
}

struct TreeMutator {
    immutable: SyntaxNode,
    mutable_clone: SyntaxNode,
}

impl TreeMutator {
    fn new(immutable: &SyntaxNode) -> TreeMutator {
        let immutable = immutable.clone();
        let mutable_clone = immutable.clone_for_update();
        TreeMutator { immutable, mutable_clone }
    }

    fn make_element_mut(&self, element: &SyntaxElement) -> SyntaxElement {
        match element {
            SyntaxElement::Node(node) => SyntaxElement::Node(self.make_syntax_mut(node)),
            SyntaxElement::Token(token) => {
                let parent = self.make_syntax_mut(&token.parent().unwrap());
                parent.children_with_tokens().nth(token.index()).unwrap()
            }
        }
    }

    fn make_syntax_mut(&self, node: &SyntaxNode) -> SyntaxNode {
        let ptr = SyntaxNodePtr::new(node);
        ptr.to_node(&self.mutable_clone)
    }
}
