//! Maps syntax elements through disjoint syntax nodes.
//!
//! [`SyntaxMappingBuilder`] should be used to create mappings to add to a [`SyntaxEditor`]

use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::{SyntaxElement, SyntaxNode};

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
    /// Like [`SyntaxMapping::upmap_child`] but for syntax elements.
    pub fn upmap_child_element(
        &self,
        child: &SyntaxElement,
        input_ancestor: &SyntaxNode,
        output_ancestor: &SyntaxNode,
    ) -> Result<SyntaxElement, MissingMapping> {
        match child {
            SyntaxElement::Node(node) => {
                self.upmap_child(node, input_ancestor, output_ancestor).map(SyntaxElement::Node)
            }
            SyntaxElement::Token(token) => {
                let upmap_parent =
                    self.upmap_child(&token.parent().unwrap(), input_ancestor, output_ancestor)?;

                let element = upmap_parent.children_with_tokens().nth(token.index()).unwrap();
                debug_assert!(
                    element.as_token().is_some_and(|it| it.kind() == token.kind()),
                    "token upmapping mapped to the wrong node ({token:?} -> {element:?})"
                );

                Ok(element)
            }
        }
    }

    /// Maps a child node of the input ancestor to the corresponding node in
    /// the output ancestor.
    pub fn upmap_child(
        &self,
        child: &SyntaxNode,
        input_ancestor: &SyntaxNode,
        output_ancestor: &SyntaxNode,
    ) -> Result<SyntaxNode, MissingMapping> {
        debug_assert!(
            child == input_ancestor
                || child.ancestors().any(|ancestor| &ancestor == input_ancestor)
        );

        // Build a list mapping up to the first mappable ancestor
        let to_first_upmap = if child != input_ancestor {
            std::iter::successors(Some((child.index(), child.clone())), |(_, current)| {
                let parent = current.parent().unwrap();

                if &parent == input_ancestor {
                    return None;
                }

                Some((parent.index(), parent))
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
        } else {
            vec![]
        };

        // Progressively up-map the input ancestor until we get to the output ancestor
        let to_output_ancestor = if input_ancestor != output_ancestor {
            self.upmap_to_ancestor(input_ancestor, output_ancestor)?
        } else {
            vec![]
        };

        let to_map_down =
            to_output_ancestor.into_iter().rev().chain(to_first_upmap.into_iter().rev());

        let mut target = output_ancestor.clone();

        for index in to_map_down {
            target = target
                .children_with_tokens()
                .nth(index)
                .and_then(|it| it.into_node())
                .expect("equivalent ancestor node should be present in target tree");
        }

        debug_assert_eq!(child.kind(), target.kind());

        Ok(target)
    }

    fn upmap_to_ancestor(
        &self,
        input_ancestor: &SyntaxNode,
        output_ancestor: &SyntaxNode,
    ) -> Result<Vec<usize>, MissingMapping> {
        let mut current =
            self.upmap_node_single(input_ancestor).unwrap_or_else(|| input_ancestor.clone());
        let mut upmap_chain = vec![current.index()];

        loop {
            let Some(parent) = current.parent() else { break };

            if &parent == output_ancestor {
                return Ok(upmap_chain);
            }

            current = match self.upmap_node_single(&parent) {
                Some(next) => next,
                None => parent,
            };
            upmap_chain.push(current.index());
        }

        Err(MissingMapping(current))
    }

    pub fn upmap_element(
        &self,
        input: &SyntaxElement,
        output_root: &SyntaxNode,
    ) -> Option<Result<SyntaxElement, MissingMapping>> {
        match input {
            SyntaxElement::Node(node) => {
                Some(self.upmap_node(node, output_root)?.map(SyntaxElement::Node))
            }
            SyntaxElement::Token(token) => {
                let upmap_parent = match self.upmap_node(&token.parent().unwrap(), output_root)? {
                    Ok(it) => it,
                    Err(err) => return Some(Err(err)),
                };

                let element = upmap_parent.children_with_tokens().nth(token.index()).unwrap();
                debug_assert!(
                    element.as_token().is_some_and(|it| it.kind() == token.kind()),
                    "token upmapping mapped to the wrong node ({token:?} -> {element:?})"
                );

                Some(Ok(element))
            }
        }
    }

    pub fn upmap_node(
        &self,
        input: &SyntaxNode,
        output_root: &SyntaxNode,
    ) -> Option<Result<SyntaxNode, MissingMapping>> {
        // Try to follow the mapping tree, if it exists
        let input_mapping = self.upmap_node_single(input);
        let input_ancestor =
            input.ancestors().find_map(|ancestor| self.upmap_node_single(&ancestor));

        match (input_mapping, input_ancestor) {
            (Some(input_mapping), _) => {
                // A mapping exists at the input, follow along the tree
                Some(self.upmap_child(&input_mapping, &input_mapping, output_root))
            }
            (None, Some(input_ancestor)) => {
                // A mapping exists at an ancestor, follow along the tree
                Some(self.upmap_child(input, &input_ancestor, output_root))
            }
            (None, None) => {
                // No mapping exists at all, is the same position in the final tree
                None
            }
        }
    }

    pub fn merge(&mut self, mut other: SyntaxMapping) {
        // Remap other's entry parents to be after the current list of entry parents
        let remap_base: u32 = self.entry_parents.len().try_into().unwrap();

        self.entry_parents.append(&mut other.entry_parents);
        self.node_mappings.extend(other.node_mappings.into_iter().map(|(node, entry)| {
            (node, MappingEntry { parent: entry.parent + remap_base, ..entry })
        }));
    }

    /// Follows the input one step along the syntax mapping tree
    fn upmap_node_single(&self, input: &SyntaxNode) -> Option<SyntaxNode> {
        let MappingEntry { parent, child_slot } = self.node_mappings.get(input)?;

        let output = self.entry_parents[*parent as usize]
            .children_with_tokens()
            .nth(*child_slot as usize)
            .and_then(SyntaxElement::into_node)
            .unwrap();

        debug_assert_eq!(input.kind(), output.kind());
        Some(output)
    }

    pub fn add_mapping(&mut self, syntax_mapping: SyntaxMappingBuilder) {
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
        input: impl IntoIterator<Item = SyntaxNode>,
        output: impl IntoIterator<Item = SyntaxNode>,
    ) {
        for pairs in input.into_iter().zip_longest(output) {
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

    pub fn finish(self, mappings: &mut SyntaxMapping) {
        mappings.add_mapping(self);
    }
}

#[derive(Debug)]
pub struct MissingMapping(pub SyntaxNode);

#[derive(Debug, Clone, Copy)]
struct MappingEntry {
    parent: u32,
    child_slot: u32,
}
