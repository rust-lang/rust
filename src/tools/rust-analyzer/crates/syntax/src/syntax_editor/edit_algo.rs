use std::{collections::VecDeque, ops::RangeInclusive};

use rowan::TextRange;

use crate::{
    syntax_editor::{Change, ChangeKind},
    ted, SyntaxElement, SyntaxNode, SyntaxNodePtr,
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

    dbg!(("initial: ", &root));
    dbg!(&changes);

    // Sort changes by range then change kind, so that we can:
    // - ensure that parent edits are ordered before child edits
    // - ensure that inserts will be guaranteed to be inserted at the right range
    // - easily check for disjoint replace ranges
    changes.sort_by(|a, b| {
        a.target_range()
            .start()
            .cmp(&b.target_range().start())
            .then(a.change_kind().cmp(&b.change_kind()))
    });

    let disjoint_replaces_ranges = changes.iter().zip(changes.iter().skip(1)).all(|(l, r)| {
        l.change_kind() == ChangeKind::Replace
            && r.change_kind() == ChangeKind::Replace
            && (l.target_parent() != r.target_parent()
                || l.target_range().intersect(r.target_range()).is_none())
    });

    if stdx::never!(
        !disjoint_replaces_ranges,
        "some replace change ranges intersect: {:?}",
        changes
    ) {
        return SyntaxEdit { root, annotations: Default::default(), changed_elements: vec![] };
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
            Change::Replace(target, _) => {
                changed_ancestors.push_back(ChangedAncestor::single(target, change_index))
            }
        }
    }

    dbg!(("before: ", &changes, &dependent_changes, &independent_changes));

    // Map change targets to the correct syntax nodes
    let tree_mutator = TreeMutator::new(&root);

    for index in independent_changes {
        match &mut changes[index as usize] {
            Change::Replace(target, _) => {
                *target = tree_mutator.make_element_mut(target);
            }
        }
    }

    for DependentChange { parent, child } in dependent_changes.into_iter() {
        let (input_ancestor, output_ancestor) = match &changes[parent as usize] {
            // insert? unreachable
            Change::Replace(target, Some(new_target)) => {
                (to_owning_node(target), to_owning_node(new_target))
            }
            Change::Replace(_, None) => continue, // silently drop outdated change
        };

        match &mut changes[child as usize] {
            Change::Replace(target, _) => {
                *target = mappings.upmap_child_element(target, &input_ancestor, output_ancestor)
            }
        }
    }

    dbg!(("after: ", &changes));

    // Apply changes
    for change in changes {
        match change {
            Change::Replace(target, None) => ted::remove(target),
            Change::Replace(target, Some(new_target)) => ted::replace(target, new_target),
        }
    }

    dbg!(("modified:", tree_mutator.mutable_clone));

    todo!("draw the rest of the owl")
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
    Range { changed_nodes: RangeInclusive<SyntaxNode>, in_parent: SyntaxNode },
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

    fn affected_range(&self) -> TextRange {
        match &self.kind {
            ChangedAncestorKind::Single { node } => node.text_range(),
            ChangedAncestorKind::Range { changed_nodes, in_parent: _ } => TextRange::new(
                changed_nodes.start().text_range().start(),
                changed_nodes.end().text_range().end(),
            ),
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
            SyntaxElement::Node(node) => SyntaxElement::Node(self.make_syntax_mut(&node)),
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
