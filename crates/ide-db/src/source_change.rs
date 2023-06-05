//! This modules defines type to represent changes to the source code, that flow
//! from the server to the client.
//!
//! It can be viewed as a dual for `Change`.

use std::{collections::hash_map::Entry, iter, mem};

use crate::SnippetCap;
use base_db::{AnchoredPathBuf, FileId};
use nohash_hasher::IntMap;
use stdx::never;
use syntax::{algo, ast, ted, AstNode, SyntaxNode, SyntaxNodePtr, TextRange, TextSize};
use text_edit::{TextEdit, TextEditBuilder};

#[derive(Default, Debug, Clone)]
pub struct SourceChange {
    pub source_file_edits: IntMap<FileId, TextEdit>,
    pub file_system_edits: Vec<FileSystemEdit>,
    pub is_snippet: bool,
}

impl SourceChange {
    /// Creates a new SourceChange with the given label
    /// from the edits.
    pub fn from_edits(
        source_file_edits: IntMap<FileId, TextEdit>,
        file_system_edits: Vec<FileSystemEdit>,
    ) -> Self {
        SourceChange { source_file_edits, file_system_edits, is_snippet: false }
    }

    pub fn from_text_edit(file_id: FileId, edit: TextEdit) -> Self {
        SourceChange {
            source_file_edits: iter::once((file_id, edit)).collect(),
            ..Default::default()
        }
    }

    /// Inserts a [`TextEdit`] for the given [`FileId`]. This properly handles merging existing
    /// edits for a file if some already exist.
    pub fn insert_source_edit(&mut self, file_id: FileId, edit: TextEdit) {
        match self.source_file_edits.entry(file_id) {
            Entry::Occupied(mut entry) => {
                never!(entry.get_mut().union(edit).is_err(), "overlapping edits for same file");
            }
            Entry::Vacant(entry) => {
                entry.insert(edit);
            }
        }
    }

    pub fn push_file_system_edit(&mut self, edit: FileSystemEdit) {
        self.file_system_edits.push(edit);
    }

    pub fn get_source_edit(&self, file_id: FileId) -> Option<&TextEdit> {
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
        iter.into_iter().for_each(|(file_id, edit)| self.insert_source_edit(file_id, edit));
    }
}

impl Extend<FileSystemEdit> for SourceChange {
    fn extend<T: IntoIterator<Item = FileSystemEdit>>(&mut self, iter: T) {
        iter.into_iter().for_each(|edit| self.push_file_system_edit(edit));
    }
}

impl From<IntMap<FileId, TextEdit>> for SourceChange {
    fn from(source_file_edits: IntMap<FileId, TextEdit>) -> SourceChange {
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
        // Render snippets first so that they get bundled into the tree diff
        if let Some(mut snippets) = self.snippet_builder.take() {
            // Last snippet always has stop index 0
            let last_stop = snippets.places.pop().unwrap();
            last_stop.place(0);

            for (index, stop) in snippets.places.into_iter().enumerate() {
                stop.place(index + 1)
            }
        }

        if let Some(tm) = self.mutated_tree.take() {
            algo::diff(&tm.immutable, &tm.mutable_clone).into_text_edit(&mut self.edit)
        }

        let edit = mem::take(&mut self.edit).finish();
        if !edit.is_empty() {
            self.source_change.insert_source_edit(self.file_id, edit);
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
        self.add_snippet(PlaceSnippet::Before(node.syntax().clone()));
    }

    /// Adds a tabstop snippet to place the cursor after `node`
    pub fn add_tabstop_after(&mut self, _cap: SnippetCap, node: impl AstNode) {
        assert!(node.syntax().parent().is_some());
        self.add_snippet(PlaceSnippet::After(node.syntax().clone()));
    }

    /// Adds a snippet to move the cursor selected over `node`
    pub fn add_placeholder_snippet(&mut self, _cap: SnippetCap, node: impl AstNode) {
        assert!(node.syntax().parent().is_some());
        self.add_snippet(PlaceSnippet::Over(node.syntax().clone()))
    }

    fn add_snippet(&mut self, snippet: PlaceSnippet) {
        let snippet_builder = self.snippet_builder.get_or_insert(SnippetBuilder { places: vec![] });
        snippet_builder.places.push(snippet);
        self.source_change.is_snippet = true;
    }

    pub fn finish(mut self) -> SourceChange {
        self.commit();
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

enum PlaceSnippet {
    /// Place a tabstop before a node
    Before(SyntaxNode),
    /// Place a tabstop before a node
    After(SyntaxNode),
    /// Place a placeholder snippet in place of the node
    Over(SyntaxNode),
}

impl PlaceSnippet {
    /// Places the snippet before or over a node with the given tab stop index
    fn place(self, order: usize) {
        // ensure the target node is still attached
        match &self {
            PlaceSnippet::Before(node) | PlaceSnippet::After(node) | PlaceSnippet::Over(node) => {
                // node should still be in the tree, but if it isn't
                // then it's okay to just ignore this place
                if stdx::never!(node.parent().is_none()) {
                    return;
                }
            }
        }

        match self {
            PlaceSnippet::Before(node) => {
                ted::insert_raw(ted::Position::before(&node), Self::make_tab_stop(order));
            }
            PlaceSnippet::After(node) => {
                ted::insert_raw(ted::Position::after(&node), Self::make_tab_stop(order));
            }
            PlaceSnippet::Over(node) => {
                let position = ted::Position::before(&node);
                node.detach();

                let snippet = ast::SourceFile::parse(&format!("${{{order}:_}}"))
                    .syntax_node()
                    .clone_for_update();

                let placeholder =
                    snippet.descendants().find_map(ast::UnderscoreExpr::cast).unwrap();
                ted::replace(placeholder.syntax(), node);

                ted::insert_raw(position, snippet);
            }
        }
    }

    fn make_tab_stop(order: usize) -> SyntaxNode {
        let stop = ast::SourceFile::parse(&format!("stop!(${order})"))
            .syntax_node()
            .descendants()
            .find_map(ast::TokenTree::cast)
            .unwrap()
            .syntax()
            .clone_for_update();

        stop.first_token().unwrap().detach();
        stop.last_token().unwrap().detach();

        stop
    }
}
