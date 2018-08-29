use relative_path::RelativePathBuf;
use libsyntax2::{File, TextRange, TextUnit, AtomEdit};
use libeditor;
use {imp::AnalysisImpl, FileId, Query};

pub use libeditor::{
    LocalEdit, StructureNode, LineIndex, FileSymbol,
    Runnable, RunnableKind, HighlightedRange, CompletionItem
};

#[derive(Debug)]
pub struct SourceChange {
    pub label: String,
    pub source_file_edits: Vec<SourceFileEdit>,
    pub file_system_edits: Vec<FileSystemEdit>,
    pub cursor_position: Option<Position>,
}

#[derive(Debug)]
pub struct Position {
    pub file_id: FileId,
    pub offset: TextUnit,
}

#[derive(Debug)]
pub struct SourceFileEdit {
    pub file_id: FileId,
    pub edits: Vec<AtomEdit>,
}

#[derive(Debug)]
pub enum FileSystemEdit {
    CreateFile {
        anchor: FileId,
        path: RelativePathBuf,
    },
    MoveFile {
        file: FileId,
        path: RelativePathBuf,
    }
}

#[derive(Debug)]
pub struct Diagnostic {
    pub message: String,
    pub range: TextRange,
    pub fix: Option<SourceChange>,
}

#[derive(Clone, Debug)]
pub struct Analysis {
    pub(crate) imp: AnalysisImpl
}

impl Analysis {
    pub fn file_syntax(&self, file_id: FileId) -> File {
        self.imp.file_syntax(file_id)
    }
    pub fn file_line_index(&self, file_id: FileId) -> LineIndex {
        self.imp.file_line_index(file_id)
    }
    pub fn extend_selection(&self, file: &File, range: TextRange) -> TextRange {
        libeditor::extend_selection(file, range).unwrap_or(range)
    }
    pub fn matching_brace(&self, file: &File, offset: TextUnit) -> Option<TextUnit> {
        libeditor::matching_brace(file, offset)
    }
    pub fn syntax_tree(&self, file_id: FileId) -> String {
        let file = self.file_syntax(file_id);
        libeditor::syntax_tree(&file)
    }
    pub fn join_lines(&self, file_id: FileId, range: TextRange) -> SourceChange {
        let file = self.file_syntax(file_id);
        SourceChange::from_local_edit(file_id, "join lines", libeditor::join_lines(&file, range))
    }
    pub fn on_eq_typed(&self, file_id: FileId, offset: TextUnit) -> Option<SourceChange> {
        let file = self.file_syntax(file_id);
        Some(SourceChange::from_local_edit(file_id, "add semicolon", libeditor::on_eq_typed(&file, offset)?))
    }
    pub fn file_structure(&self, file_id: FileId) -> Vec<StructureNode> {
        let file = self.file_syntax(file_id);
        libeditor::file_structure(&file)
    }
    pub fn symbol_search(&self, query: Query) -> Vec<(FileId, FileSymbol)> {
        self.imp.world_symbols(query)
    }
    pub fn approximately_resolve_symbol(&self, file_id: FileId, offset: TextUnit) -> Vec<(FileId, FileSymbol)> {
        self.imp.approximately_resolve_symbol(file_id, offset)
    }
    pub fn parent_module(&self, file_id: FileId) -> Vec<(FileId, FileSymbol)> {
        self.imp.parent_module(file_id)
    }
    pub fn runnables(&self, file_id: FileId) -> Vec<Runnable> {
        let file = self.file_syntax(file_id);
        libeditor::runnables(&file)
    }
    pub fn highlight(&self, file_id: FileId) -> Vec<HighlightedRange> {
        let file = self.file_syntax(file_id);
        libeditor::highlight(&file)
    }
    pub fn completions(&self, file_id: FileId, offset: TextUnit) -> Option<Vec<CompletionItem>> {
        let file = self.file_syntax(file_id);
        libeditor::scope_completion(&file, offset)
    }
    pub fn assists(&self, file_id: FileId, offset: TextUnit) -> Vec<SourceChange> {
        self.imp.assists(file_id, offset)
    }
    pub fn diagnostics(&self, file_id: FileId) -> Vec<Diagnostic> {
        self.imp.diagnostics(file_id)
    }
}

impl SourceChange {
    pub(crate) fn from_local_edit(file_id: FileId, label: &str, edit: LocalEdit) -> SourceChange {
        let file_edit = SourceFileEdit {
            file_id,
            edits: edit.edit.into_atoms(),
        };
        SourceChange {
            label: label.to_string(),
            source_file_edits: vec![file_edit],
            file_system_edits: vec![],
            cursor_position: edit.cursor_position
                .map(|offset| Position { offset, file_id })
        }
    }
}
