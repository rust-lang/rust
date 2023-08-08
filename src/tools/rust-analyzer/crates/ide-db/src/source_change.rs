//! This modules defines type to represent changes to the source code, that flow
//! from the server to the client.
//!
//! It can be viewed as a dual for `Change`.

use std::{collections::hash_map::Entry, iter, mem};

use crate::SnippetCap;
use base_db::{AnchoredPathBuf, FileId};
use itertools::Itertools;
use nohash_hasher::IntMap;
use stdx::never;
use syntax::{
    algo, AstNode, SyntaxElement, SyntaxNode, SyntaxNodePtr, SyntaxToken, TextRange, TextSize,
};
use text_edit::{TextEdit, TextEditBuilder};

#[derive(Default, Debug, Clone)]
pub struct SourceChange {
    pub source_file_edits: IntMap<FileId, (TextEdit, Option<SnippetEdit>)>,
    pub file_system_edits: Vec<FileSystemEdit>,
    pub is_snippet: bool,
}

impl SourceChange {
    /// Creates a new SourceChange with the given label
    /// from the edits.
    pub fn from_edits(
        source_file_edits: IntMap<FileId, (TextEdit, Option<SnippetEdit>)>,
        file_system_edits: Vec<FileSystemEdit>,
    ) -> Self {
        SourceChange { source_file_edits, file_system_edits, is_snippet: false }
    }

    pub fn from_text_edit(file_id: FileId, edit: TextEdit) -> Self {
        SourceChange {
            source_file_edits: iter::once((file_id, (edit, None))).collect(),
            ..Default::default()
        }
    }

    /// Inserts a [`TextEdit`] for the given [`FileId`]. This properly handles merging existing
    /// edits for a file if some already exist.
    pub fn insert_source_edit(&mut self, file_id: FileId, edit: TextEdit) {
        self.insert_source_and_snippet_edit(file_id, edit, None)
    }

    /// Inserts a [`TextEdit`] and potentially a [`SnippetEdit`] for the given [`FileId`].
    /// This properly handles merging existing edits for a file if some already exist.
    pub fn insert_source_and_snippet_edit(
        &mut self,
        file_id: FileId,
        edit: TextEdit,
        snippet_edit: Option<SnippetEdit>,
    ) {
        match self.source_file_edits.entry(file_id) {
            Entry::Occupied(mut entry) => {
                let value = entry.get_mut();
                never!(value.0.union(edit).is_err(), "overlapping edits for same file");
                never!(
                    value.1.is_some() && snippet_edit.is_some(),
                    "overlapping snippet edits for same file"
                );
                if value.1.is_none() {
                    value.1 = snippet_edit;
                }
            }
            Entry::Vacant(entry) => {
                entry.insert((edit, snippet_edit));
            }
        }
    }

    pub fn push_file_system_edit(&mut self, edit: FileSystemEdit) {
        self.file_system_edits.push(edit);
    }

    pub fn get_source_and_snippet_edit(
        &self,
        file_id: FileId,
    ) -> Option<&(TextEdit, Option<SnippetEdit>)> {
        self.source_file_edits.get(&file_id)
    }

    pub fn merge(mut self, other: SourceChange) -> SourceChange {
        self.extend(other.source_file_edits);
        self.extend(other.file_system_edits);
        self.is_snippet |= other.is_snippet;
        self
    }
}

impl Extend<(FileId, TextEdit)> for SourceChange {
    fn extend<T: IntoIterator<Item = (FileId, TextEdit)>>(&mut self, iter: T) {
        self.extend(iter.into_iter().map(|(file_id, edit)| (file_id, (edit, None))))
    }
}

impl Extend<(FileId, (TextEdit, Option<SnippetEdit>))> for SourceChange {
    fn extend<T: IntoIterator<Item = (FileId, (TextEdit, Option<SnippetEdit>))>>(
        &mut self,
        iter: T,
    ) {
        iter.into_iter().for_each(|(file_id, (edit, snippet_edit))| {
            self.insert_source_and_snippet_edit(file_id, edit, snippet_edit)
        });
    }
}

impl Extend<FileSystemEdit> for SourceChange {
    fn extend<T: IntoIterator<Item = FileSystemEdit>>(&mut self, iter: T) {
        iter.into_iter().for_each(|edit| self.push_file_system_edit(edit));
    }
}

impl From<IntMap<FileId, TextEdit>> for SourceChange {
    fn from(source_file_edits: IntMap<FileId, TextEdit>) -> SourceChange {
        let source_file_edits =
            source_file_edits.into_iter().map(|(file_id, edit)| (file_id, (edit, None))).collect();
        SourceChange { source_file_edits, file_system_edits: Vec::new(), is_snippet: false }
    }
}

impl FromIterator<(FileId, TextEdit)> for SourceChange {
    fn from_iter<T: IntoIterator<Item = (FileId, TextEdit)>>(iter: T) -> Self {
        let mut this = SourceChange::default();
        this.extend(iter);
        this
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnippetEdit(Vec<(u32, TextRange)>);

impl SnippetEdit {
    pub fn new(snippets: Vec<Snippet>) -> Self {
        let mut snippet_ranges = snippets
            .into_iter()
            .zip(1..)
            .with_position()
            .map(|pos| {
                let (snippet, index) = match pos {
                    itertools::Position::First(it) | itertools::Position::Middle(it) => it,
                    // last/only snippet gets index 0
                    itertools::Position::Last((snippet, _))
                    | itertools::Position::Only((snippet, _)) => (snippet, 0),
                };

                let range = match snippet {
                    Snippet::Tabstop(pos) => TextRange::empty(pos),
                    Snippet::Placeholder(range) => range,
                };
                (index, range)
            })
            .collect_vec();

        snippet_ranges.sort_by_key(|(_, range)| range.start());

        // Ensure that none of the ranges overlap
        let disjoint_ranges = snippet_ranges
            .iter()
            .zip(snippet_ranges.iter().skip(1))
            .all(|((_, left), (_, right))| left.end() <= right.start() || left == right);
        stdx::always!(disjoint_ranges);

        SnippetEdit(snippet_ranges)
    }

    /// Inserts all of the snippets into the given text.
    pub fn apply(&self, text: &mut String) {
        // Start from the back so that we don't have to adjust ranges
        for (index, range) in self.0.iter().rev() {
            if range.is_empty() {
                // is a tabstop
                text.insert_str(range.start().into(), &format!("${index}"));
            } else {
                // is a placeholder
                text.insert(range.end().into(), '}');
                text.insert_str(range.start().into(), &format!("${{{index}:"));
            }
        }
    }

    /// Gets the underlying snippet index + text range
    /// Tabstops are represented by an empty range, and placeholders use the range that they were given
    pub fn into_edit_ranges(self) -> Vec<(u32, TextRange)> {
        self.0
    }
}

pub struct SourceChangeBuilder {
    pub edit: TextEditBuilder,
    pub file_id: FileId,
    pub source_change: SourceChange,
    pub trigger_signature_help: bool,

    /// Maps the original, immutable `SyntaxNode` to a `clone_for_update` twin.
    pub mutated_tree: Option<TreeMutator>,
    /// Keeps track of where to place snippets
    pub snippet_builder: Option<SnippetBuilder>,
}

pub struct TreeMutator {
    immutable: SyntaxNode,
    mutable_clone: SyntaxNode,
}

#[derive(Default)]
pub struct SnippetBuilder {
    /// Where to place snippets at
    places: Vec<PlaceSnippet>,
}

impl TreeMutator {
    pub fn new(immutable: &SyntaxNode) -> TreeMutator {
        let immutable = immutable.ancestors().last().unwrap();
        let mutable_clone = immutable.clone_for_update();
        TreeMutator { immutable, mutable_clone }
    }

    pub fn make_mut<N: AstNode>(&self, node: &N) -> N {
        N::cast(self.make_syntax_mut(node.syntax())).unwrap()
    }

    pub fn make_syntax_mut(&self, node: &SyntaxNode) -> SyntaxNode {
        let ptr = SyntaxNodePtr::new(node);
        ptr.to_node(&self.mutable_clone)
    }
}

impl SourceChangeBuilder {
    pub fn new(file_id: FileId) -> SourceChangeBuilder {
        SourceChangeBuilder {
            edit: TextEdit::builder(),
            file_id,
            source_change: SourceChange::default(),
            trigger_signature_help: false,
            mutated_tree: None,
            snippet_builder: None,
        }
    }

    pub fn edit_file(&mut self, file_id: FileId) {
        self.commit();
        self.file_id = file_id;
    }

    fn commit(&mut self) {
        let snippet_edit = self.snippet_builder.take().map(|builder| {
            SnippetEdit::new(
                builder.places.into_iter().map(PlaceSnippet::finalize_position).collect_vec(),
            )
        });

        if let Some(tm) = self.mutated_tree.take() {
            algo::diff(&tm.immutable, &tm.mutable_clone).into_text_edit(&mut self.edit);
        }

        let edit = mem::take(&mut self.edit).finish();
        if !edit.is_empty() || snippet_edit.is_some() {
            self.source_change.insert_source_and_snippet_edit(self.file_id, edit, snippet_edit);
        }
    }

    pub fn make_mut<N: AstNode>(&mut self, node: N) -> N {
        self.mutated_tree.get_or_insert_with(|| TreeMutator::new(node.syntax())).make_mut(&node)
    }
    /// Returns a copy of the `node`, suitable for mutation.
    ///
    /// Syntax trees in rust-analyzer are typically immutable, and mutating
    /// operations panic at runtime. However, it is possible to make a copy of
    /// the tree and mutate the copy freely. Mutation is based on interior
    /// mutability, and different nodes in the same tree see the same mutations.
    ///
    /// The typical pattern for an assist is to find specific nodes in the read
    /// phase, and then get their mutable counterparts using `make_mut` in the
    /// mutable state.
    pub fn make_syntax_mut(&mut self, node: SyntaxNode) -> SyntaxNode {
        self.mutated_tree.get_or_insert_with(|| TreeMutator::new(&node)).make_syntax_mut(&node)
    }

    /// Remove specified `range` of text.
    pub fn delete(&mut self, range: TextRange) {
        self.edit.delete(range)
    }
    /// Append specified `text` at the given `offset`
    pub fn insert(&mut self, offset: TextSize, text: impl Into<String>) {
        self.edit.insert(offset, text.into())
    }
    /// Append specified `snippet` at the given `offset`
    pub fn insert_snippet(
        &mut self,
        _cap: SnippetCap,
        offset: TextSize,
        snippet: impl Into<String>,
    ) {
        self.source_change.is_snippet = true;
        self.insert(offset, snippet);
    }
    /// Replaces specified `range` of text with a given string.
    pub fn replace(&mut self, range: TextRange, replace_with: impl Into<String>) {
        self.edit.replace(range, replace_with.into())
    }
    /// Replaces specified `range` of text with a given `snippet`.
    pub fn replace_snippet(
        &mut self,
        _cap: SnippetCap,
        range: TextRange,
        snippet: impl Into<String>,
    ) {
        self.source_change.is_snippet = true;
        self.replace(range, snippet);
    }
    pub fn replace_ast<N: AstNode>(&mut self, old: N, new: N) {
        algo::diff(old.syntax(), new.syntax()).into_text_edit(&mut self.edit)
    }
    pub fn create_file(&mut self, dst: AnchoredPathBuf, content: impl Into<String>) {
        let file_system_edit = FileSystemEdit::CreateFile { dst, initial_contents: content.into() };
        self.source_change.push_file_system_edit(file_system_edit);
    }
    pub fn move_file(&mut self, src: FileId, dst: AnchoredPathBuf) {
        let file_system_edit = FileSystemEdit::MoveFile { src, dst };
        self.source_change.push_file_system_edit(file_system_edit);
    }
    pub fn trigger_signature_help(&mut self) {
        self.trigger_signature_help = true;
    }

    /// Adds a tabstop snippet to place the cursor before `node`
    pub fn add_tabstop_before(&mut self, _cap: SnippetCap, node: impl AstNode) {
        assert!(node.syntax().parent().is_some());
        self.add_snippet(PlaceSnippet::Before(node.syntax().clone().into()));
    }

    /// Adds a tabstop snippet to place the cursor after `node`
    pub fn add_tabstop_after(&mut self, _cap: SnippetCap, node: impl AstNode) {
        assert!(node.syntax().parent().is_some());
        self.add_snippet(PlaceSnippet::After(node.syntax().clone().into()));
    }

    /// Adds a tabstop snippet to place the cursor before `token`
    pub fn add_tabstop_before_token(&mut self, _cap: SnippetCap, token: SyntaxToken) {
        assert!(token.parent().is_some());
        self.add_snippet(PlaceSnippet::Before(token.clone().into()));
    }

    /// Adds a tabstop snippet to place the cursor after `token`
    pub fn add_tabstop_after_token(&mut self, _cap: SnippetCap, token: SyntaxToken) {
        assert!(token.parent().is_some());
        self.add_snippet(PlaceSnippet::After(token.clone().into()));
    }

    /// Adds a snippet to move the cursor selected over `node`
    pub fn add_placeholder_snippet(&mut self, _cap: SnippetCap, node: impl AstNode) {
        assert!(node.syntax().parent().is_some());
        self.add_snippet(PlaceSnippet::Over(node.syntax().clone().into()))
    }

    fn add_snippet(&mut self, snippet: PlaceSnippet) {
        let snippet_builder = self.snippet_builder.get_or_insert(SnippetBuilder { places: vec![] });
        snippet_builder.places.push(snippet);
        self.source_change.is_snippet = true;
    }

    pub fn finish(mut self) -> SourceChange {
        self.commit();

        // Only one file can have snippet edits
        stdx::never!(self
            .source_change
            .source_file_edits
            .iter()
            .filter(|(_, (_, snippet_edit))| snippet_edit.is_some())
            .at_most_one()
            .is_err());

        mem::take(&mut self.source_change)
    }
}

#[derive(Debug, Clone)]
pub enum FileSystemEdit {
    CreateFile { dst: AnchoredPathBuf, initial_contents: String },
    MoveFile { src: FileId, dst: AnchoredPathBuf },
    MoveDir { src: AnchoredPathBuf, src_id: FileId, dst: AnchoredPathBuf },
}

impl From<FileSystemEdit> for SourceChange {
    fn from(edit: FileSystemEdit) -> SourceChange {
        SourceChange {
            source_file_edits: Default::default(),
            file_system_edits: vec![edit],
            is_snippet: false,
        }
    }
}

pub enum Snippet {
    /// A tabstop snippet (e.g. `$0`).
    Tabstop(TextSize),
    /// A placeholder snippet (e.g. `${0:placeholder}`).
    Placeholder(TextRange),
}

enum PlaceSnippet {
    /// Place a tabstop before an element
    Before(SyntaxElement),
    /// Place a tabstop before an element
    After(SyntaxElement),
    /// Place a placeholder snippet in place of the element
    Over(SyntaxElement),
}

impl PlaceSnippet {
    fn finalize_position(self) -> Snippet {
        match self {
            PlaceSnippet::Before(it) => Snippet::Tabstop(it.text_range().start()),
            PlaceSnippet::After(it) => Snippet::Tabstop(it.text_range().end()),
            PlaceSnippet::Over(it) => Snippet::Placeholder(it.text_range()),
        }
    }
}
