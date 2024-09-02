use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::{SyntaxElement, SyntaxNode};

use super::SyntaxEditor;

#[derive(Debug, Default)]
pub struct SyntaxMapping {
    // important information to keep track of:
    // node -> node
    // token -> token (implicit in mappings)
    // input parent -> output parent (for deep lookups)

    // mappings ->  parents
    entry_parents: Vec<SyntaxNode>,
    node_mappings: FxHashMap<SyntaxNode, MappingEntry>,
}

impl SyntaxMapping {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upmap_child_element(
        &self,
        child: &SyntaxElement,
        input_ancestor: &SyntaxNode,
        output_ancestor: SyntaxNode,
    ) -> SyntaxElement {
        match child {
            SyntaxElement::Node(node) => {
                SyntaxElement::Node(self.upmap_child(node, input_ancestor, output_ancestor))
            }
            SyntaxElement::Token(token) => {
                let upmap_parent =
                    self.upmap_child(&token.parent().unwrap(), input_ancestor, output_ancestor);

                let element = upmap_parent.children_with_tokens().nth(token.index()).unwrap();
                debug_assert!(
                    element.as_token().is_some_and(|it| it.kind() == token.kind()),
                    "token upmapping mapped to the wrong node ({token:?} -> {element:?})"
                );

                element
            }
        }
    }

    pub fn upmap_child(
        &self,
        child: &SyntaxNode,
        input_ancestor: &SyntaxNode,
        output_ancestor: SyntaxNode,
    ) -> SyntaxNode {
        debug_assert!(child.ancestors().any(|ancestor| &ancestor == input_ancestor));

        // Build a list mapping up to the first mappable ancestor
        let to_first_upmap =
            std::iter::successors(Some((child.index(), child.clone())), |(_, current)| {
                let parent = current.parent().unwrap();

                if &parent == input_ancestor {
                    return None;
                }

                Some((parent.index(), parent))
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        // Progressively up-map the input ancestor until we get to the output ancestor
        let to_output_ancestor = if input_ancestor != &output_ancestor {
            std::iter::successors(Some((input_ancestor.index(), self.upmap_node(input_ancestor).unwrap_or_else(|| input_ancestor.clone()))), |(_,  current)| {
                let Some(parent) = current.parent() else {
                    unreachable!("no mappings exist between {current:?} (ancestor of {input_ancestor:?}) and {output_ancestor:?}")
                };

                if &parent == &output_ancestor {
                    return None;
                }

                Some((parent.index(), match self.upmap_node(&parent) {
                    Some(next) => next,
                    None => parent
                }))
            }).map(|(i, _)| i).collect::<Vec<_>>()
        } else {
            vec![]
        };

        let to_map_down =
            to_output_ancestor.into_iter().rev().chain(to_first_upmap.into_iter().rev());

        let mut target = output_ancestor;

        for index in to_map_down {
            target = target
                .children_with_tokens()
                .nth(index)
                .and_then(|it| it.into_node())
                .expect("equivalent ancestor node should be present in target tree");
        }

        debug_assert_eq!(child.kind(), target.kind());

        target
    }

    pub fn upmap_node(&self, input: &SyntaxNode) -> Option<SyntaxNode> {
        let MappingEntry { parent, child_slot } = self.node_mappings.get(input)?;

        let output = self.entry_parents[*parent as usize]
            .children_with_tokens()
            .nth(*child_slot as usize)
            .and_then(SyntaxElement::into_node)
            .unwrap();

        debug_assert_eq!(input.kind(), output.kind());
        Some(output)
    }

    pub fn merge(&mut self, mut other: SyntaxMapping) {
        // Remap other's entry parents to be after the current list of entry parents
        let remap_base: u32 = self.entry_parents.len().try_into().unwrap();

        self.entry_parents.append(&mut other.entry_parents);
        self.node_mappings.extend(other.node_mappings.into_iter().map(|(node, entry)| {
            (node, MappingEntry { parent: entry.parent + remap_base, ..entry })
        }));
    }

    fn add_mapping(&mut self, syntax_mapping: SyntaxMappingBuilder) {
        let SyntaxMappingBuilder { parent_node, node_mappings } = syntax_mapping;

        let parent_entry: u32 = self.entry_parents.len().try_into().unwrap();
        self.entry_parents.push(parent_node);

        let node_entries = node_mappings
            .into_iter()
            .map(|(node, slot)| (node, MappingEntry { parent: parent_entry, child_slot: slot }));

        self.node_mappings.extend(node_entries);
    }
}

#[derive(Debug)]
pub struct SyntaxMappingBuilder {
    parent_node: SyntaxNode,
    node_mappings: Vec<(SyntaxNode, u32)>,
}

impl SyntaxMappingBuilder {
    pub fn new(parent_node: SyntaxNode) -> Self {
        Self { parent_node, node_mappings: vec![] }
    }

    pub fn map_node(&mut self, input: SyntaxNode, output: SyntaxNode) {
        debug_assert_eq!(output.parent().as_ref(), Some(&self.parent_node));
        self.node_mappings.push((input, output.index() as u32));
    }

    pub fn map_children(
        &mut self,
        input: impl Iterator<Item = SyntaxNode>,
        output: impl Iterator<Item = SyntaxNode>,
    ) {
        for pairs in input.zip_longest(output) {
            let (input, output) = match pairs {
                itertools::EitherOrBoth::Both(l, r) => (l, r),
                itertools::EitherOrBoth::Left(_) => {
                    unreachable!("mapping more input nodes than there are output nodes")
                }
                itertools::EitherOrBoth::Right(_) => break,
            };

            self.map_node(input, output);
        }
    }

    pub fn finish(self, editor: &mut SyntaxEditor) {
        editor.mappings.add_mapping(self);
    }
}

#[derive(Debug, Clone, Copy)]
struct MappingEntry {
    parent: u32,
    child_slot: u32,
}
