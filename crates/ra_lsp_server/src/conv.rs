use lsp_types::{
    self, CreateFile, DiagnosticSeverity, DocumentChangeOperation, DocumentChanges, Documentation,
    Location, LocationLink, MarkupContent, MarkupKind, Position, Range, RenameFile, ResourceOp,
    SymbolKind, TextDocumentEdit, TextDocumentIdentifier, TextDocumentItem,
    TextDocumentPositionParams, Url, VersionedTextDocumentIdentifier, WorkspaceEdit,
};
use ra_ide_api::{
    translate_offset_with_edit, CompletionItem, CompletionItemKind, FileId, FilePosition,
    FileRange, FileSystemEdit, InsertTextFormat, LineCol, LineIndex, NavigationTarget, RangeInfo,
    Severity, SourceChange, SourceFileEdit,
};
use ra_syntax::{SyntaxKind, TextRange, TextUnit};
use ra_text_edit::{AtomTextEdit, TextEdit};

use crate::{req, world::WorldSnapshot, Result};

pub trait Conv {
    type Output;
    fn conv(self) -> Self::Output;
}

pub trait ConvWith {
    type Ctx;
    type Output;
    fn conv_with(self, ctx: &Self::Ctx) -> Self::Output;
}

pub trait TryConvWith {
    type Ctx;
    type Output;
    fn try_conv_with(self, ctx: &Self::Ctx) -> Result<Self::Output>;
}

impl Conv for SyntaxKind {
    type Output = SymbolKind;

    fn conv(self) -> <Self as Conv>::Output {
        match self {
            SyntaxKind::FN_DEF => SymbolKind::Function,
            SyntaxKind::STRUCT_DEF => SymbolKind::Struct,
            SyntaxKind::ENUM_DEF => SymbolKind::Enum,
            SyntaxKind::ENUM_VARIANT => SymbolKind::EnumMember,
            SyntaxKind::TRAIT_DEF => SymbolKind::Interface,
            SyntaxKind::MODULE => SymbolKind::Module,
            SyntaxKind::TYPE_ALIAS_DEF => SymbolKind::TypeParameter,
            SyntaxKind::NAMED_FIELD_DEF => SymbolKind::Field,
            SyntaxKind::STATIC_DEF => SymbolKind::Constant,
            SyntaxKind::CONST_DEF => SymbolKind::Constant,
            SyntaxKind::IMPL_BLOCK => SymbolKind::Object,
            _ => SymbolKind::Variable,
        }
    }
}

impl Conv for CompletionItemKind {
    type Output = ::lsp_types::CompletionItemKind;

    fn conv(self) -> <Self as Conv>::Output {
        use lsp_types::CompletionItemKind::*;
        match self {
            CompletionItemKind::Keyword => Keyword,
            CompletionItemKind::Snippet => Snippet,
            CompletionItemKind::Module => Module,
            CompletionItemKind::Function => Function,
            CompletionItemKind::Struct => Struct,
            CompletionItemKind::Enum => Enum,
            CompletionItemKind::EnumVariant => EnumMember,
            CompletionItemKind::BuiltinType => Struct,
            CompletionItemKind::Binding => Variable,
            CompletionItemKind::Field => Field,
            CompletionItemKind::Trait => Interface,
            CompletionItemKind::TypeAlias => Struct,
            CompletionItemKind::Const => Constant,
            CompletionItemKind::Static => Value,
            CompletionItemKind::Method => Method,
            CompletionItemKind::TypeParam => TypeParameter,
            CompletionItemKind::Macro => Method,
        }
    }
}

impl Conv for Severity {
    type Output = DiagnosticSeverity;
    fn conv(self) -> DiagnosticSeverity {
        match self {
            Severity::Error => DiagnosticSeverity::Error,
            Severity::WeakWarning => DiagnosticSeverity::Hint,
        }
    }
}

impl ConvWith for CompletionItem {
    type Ctx = LineIndex;
    type Output = ::lsp_types::CompletionItem;

    fn conv_with(self, ctx: &LineIndex) -> ::lsp_types::CompletionItem {
        let mut additional_text_edits = Vec::new();
        let mut text_edit = None;
        // LSP does not allow arbitrary edits in completion, so we have to do a
        // non-trivial mapping here.
        for atom_edit in self.text_edit().as_atoms() {
            if self.source_range().is_subrange(&atom_edit.delete) {
                text_edit = Some(if atom_edit.delete == self.source_range() {
                    atom_edit.conv_with(ctx)
                } else {
                    assert!(self.source_range().end() == atom_edit.delete.end());
                    let range1 =
                        TextRange::from_to(atom_edit.delete.start(), self.source_range().start());
                    let range2 = self.source_range();
                    let edit1 = AtomTextEdit::replace(range1, String::new());
                    let edit2 = AtomTextEdit::replace(range2, atom_edit.insert.clone());
                    additional_text_edits.push(edit1.conv_with(ctx));
                    edit2.conv_with(ctx)
                })
            } else {
                assert!(self.source_range().intersection(&atom_edit.delete).is_none());
                additional_text_edits.push(atom_edit.conv_with(ctx));
            }
        }
        let text_edit = text_edit.unwrap();

        let mut res = lsp_types::CompletionItem {
            label: self.label().to_string(),
            detail: self.detail().map(|it| it.to_string()),
            filter_text: Some(self.lookup().to_string()),
            kind: self.kind().map(|it| it.conv()),
            text_edit: Some(text_edit),
            additional_text_edits: Some(additional_text_edits),
            documentation: self.documentation().map(|it| it.conv()),
            ..Default::default()
        };
        res.insert_text_format = Some(match self.insert_text_format() {
            InsertTextFormat::Snippet => lsp_types::InsertTextFormat::Snippet,
            InsertTextFormat::PlainText => lsp_types::InsertTextFormat::PlainText,
        });

        res
    }
}

impl ConvWith for Position {
    type Ctx = LineIndex;
    type Output = TextUnit;

    fn conv_with(self, line_index: &LineIndex) -> TextUnit {
        let line_col = LineCol { line: self.line as u32, col_utf16: self.character as u32 };
        line_index.offset(line_col)
    }
}

impl ConvWith for TextUnit {
    type Ctx = LineIndex;
    type Output = Position;

    fn conv_with(self, line_index: &LineIndex) -> Position {
        let line_col = line_index.line_col(self);
        Position::new(u64::from(line_col.line), u64::from(line_col.col_utf16))
    }
}

impl ConvWith for TextRange {
    type Ctx = LineIndex;
    type Output = Range;

    fn conv_with(self, line_index: &LineIndex) -> Range {
        Range::new(self.start().conv_with(line_index), self.end().conv_with(line_index))
    }
}

impl ConvWith for Range {
    type Ctx = LineIndex;
    type Output = TextRange;

    fn conv_with(self, line_index: &LineIndex) -> TextRange {
        TextRange::from_to(self.start.conv_with(line_index), self.end.conv_with(line_index))
    }
}

impl Conv for ra_ide_api::Documentation {
    type Output = lsp_types::Documentation;
    fn conv(self) -> Documentation {
        Documentation::MarkupContent(MarkupContent {
            kind: MarkupKind::Markdown,
            value: crate::markdown::format_docs(self.as_str()),
        })
    }
}

impl Conv for ra_ide_api::FunctionSignature {
    type Output = lsp_types::SignatureInformation;
    fn conv(self) -> Self::Output {
        use lsp_types::{ParameterInformation, ParameterLabel, SignatureInformation};

        let label = self.to_string();

        let documentation = self.doc.map(|it| it.conv());

        let parameters: Vec<ParameterInformation> = self
            .parameters
            .into_iter()
            .map(|param| ParameterInformation {
                label: ParameterLabel::Simple(param),
                documentation: None,
            })
            .collect();

        SignatureInformation { label, documentation, parameters: Some(parameters) }
    }
}

impl ConvWith for TextEdit {
    type Ctx = LineIndex;
    type Output = Vec<lsp_types::TextEdit>;

    fn conv_with(self, line_index: &LineIndex) -> Vec<lsp_types::TextEdit> {
        self.as_atoms().iter().map_conv_with(line_index).collect()
    }
}

impl<'a> ConvWith for &'a AtomTextEdit {
    type Ctx = LineIndex;
    type Output = lsp_types::TextEdit;

    fn conv_with(self, line_index: &LineIndex) -> lsp_types::TextEdit {
        lsp_types::TextEdit {
            range: self.delete.conv_with(line_index),
            new_text: self.insert.clone(),
        }
    }
}

impl<T: ConvWith> ConvWith for Option<T> {
    type Ctx = <T as ConvWith>::Ctx;
    type Output = Option<<T as ConvWith>::Output>;
    fn conv_with(self, ctx: &Self::Ctx) -> Self::Output {
        self.map(|x| ConvWith::conv_with(x, ctx))
    }
}

impl<'a> TryConvWith for &'a Url {
    type Ctx = WorldSnapshot;
    type Output = FileId;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<FileId> {
        world.uri_to_file_id(self)
    }
}

impl TryConvWith for FileId {
    type Ctx = WorldSnapshot;
    type Output = Url;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<Url> {
        world.file_id_to_uri(self)
    }
}

impl<'a> TryConvWith for &'a TextDocumentItem {
    type Ctx = WorldSnapshot;
    type Output = FileId;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<FileId> {
        self.uri.try_conv_with(world)
    }
}

impl<'a> TryConvWith for &'a VersionedTextDocumentIdentifier {
    type Ctx = WorldSnapshot;
    type Output = FileId;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<FileId> {
        self.uri.try_conv_with(world)
    }
}

impl<'a> TryConvWith for &'a TextDocumentIdentifier {
    type Ctx = WorldSnapshot;
    type Output = FileId;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<FileId> {
        world.uri_to_file_id(&self.uri)
    }
}

impl<'a> TryConvWith for &'a TextDocumentPositionParams {
    type Ctx = WorldSnapshot;
    type Output = FilePosition;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<FilePosition> {
        let file_id = self.text_document.try_conv_with(world)?;
        let line_index = world.analysis().file_line_index(file_id)?;
        let offset = self.position.conv_with(&line_index);
        Ok(FilePosition { file_id, offset })
    }
}

impl<'a> TryConvWith for (&'a TextDocumentIdentifier, Range) {
    type Ctx = WorldSnapshot;
    type Output = FileRange;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<FileRange> {
        let file_id = self.0.try_conv_with(world)?;
        let line_index = world.analysis().file_line_index(file_id)?;
        let range = self.1.conv_with(&line_index);
        Ok(FileRange { file_id, range })
    }
}

impl<T: TryConvWith> TryConvWith for Vec<T> {
    type Ctx = <T as TryConvWith>::Ctx;
    type Output = Vec<<T as TryConvWith>::Output>;
    fn try_conv_with(self, ctx: &Self::Ctx) -> Result<Self::Output> {
        let mut res = Vec::with_capacity(self.len());
        for item in self {
            res.push(item.try_conv_with(ctx)?);
        }
        Ok(res)
    }
}

impl TryConvWith for SourceChange {
    type Ctx = WorldSnapshot;
    type Output = req::SourceChange;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<req::SourceChange> {
        let cursor_position = match self.cursor_position {
            None => None,
            Some(pos) => {
                let line_index = world.analysis().file_line_index(pos.file_id)?;
                let edit = self
                    .source_file_edits
                    .iter()
                    .find(|it| it.file_id == pos.file_id)
                    .map(|it| &it.edit);
                let line_col = match edit {
                    Some(edit) => translate_offset_with_edit(&*line_index, pos.offset, edit),
                    None => line_index.line_col(pos.offset),
                };
                let position =
                    Position::new(u64::from(line_col.line), u64::from(line_col.col_utf16));
                Some(TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier::new(pos.file_id.try_conv_with(world)?),
                    position,
                })
            }
        };
        let mut document_changes: Vec<DocumentChangeOperation> = Vec::new();
        for resource_op in self.file_system_edits.try_conv_with(world)? {
            document_changes.push(DocumentChangeOperation::Op(resource_op));
        }
        for text_document_edit in self.source_file_edits.try_conv_with(world)? {
            document_changes.push(DocumentChangeOperation::Edit(text_document_edit));
        }
        let workspace_edit = WorkspaceEdit {
            changes: None,
            document_changes: Some(DocumentChanges::Operations(document_changes)),
        };
        Ok(req::SourceChange { label: self.label, workspace_edit, cursor_position })
    }
}

impl TryConvWith for SourceFileEdit {
    type Ctx = WorldSnapshot;
    type Output = TextDocumentEdit;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<TextDocumentEdit> {
        let text_document = VersionedTextDocumentIdentifier {
            uri: self.file_id.try_conv_with(world)?,
            version: None,
        };
        let line_index = world.analysis().file_line_index(self.file_id)?;
        let edits = self.edit.as_atoms().iter().map_conv_with(&line_index).collect();
        Ok(TextDocumentEdit { text_document, edits })
    }
}

impl TryConvWith for FileSystemEdit {
    type Ctx = WorldSnapshot;
    type Output = ResourceOp;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<ResourceOp> {
        let res = match self {
            FileSystemEdit::CreateFile { source_root, path } => {
                let uri = world.path_to_uri(source_root, &path)?;
                ResourceOp::Create(CreateFile { uri, options: None })
            }
            FileSystemEdit::MoveFile { src, dst_source_root, dst_path } => {
                let old_uri = world.file_id_to_uri(src)?;
                let new_uri = world.path_to_uri(dst_source_root, &dst_path)?;
                ResourceOp::Rename(RenameFile { old_uri, new_uri, options: None })
            }
        };
        Ok(res)
    }
}

impl TryConvWith for &NavigationTarget {
    type Ctx = WorldSnapshot;
    type Output = Location;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<Location> {
        let line_index = world.analysis().file_line_index(self.file_id())?;
        let range = self.range();
        to_location(self.file_id(), range, &world, &line_index)
    }
}

impl TryConvWith for (FileId, RangeInfo<NavigationTarget>) {
    type Ctx = WorldSnapshot;
    type Output = LocationLink;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<LocationLink> {
        let (src_file_id, target) = self;

        let target_uri = target.info.file_id().try_conv_with(world)?;
        let src_line_index = world.analysis().file_line_index(src_file_id)?;
        let tgt_line_index = world.analysis().file_line_index(target.info.file_id())?;

        let target_range = target.info.full_range().conv_with(&tgt_line_index);

        let target_selection_range = target
            .info
            .focus_range()
            .map(|it| it.conv_with(&tgt_line_index))
            .unwrap_or(target_range);

        let res = LocationLink {
            origin_selection_range: Some(target.range.conv_with(&src_line_index)),
            target_uri,
            target_range,
            target_selection_range,
        };
        Ok(res)
    }
}

impl TryConvWith for (FileId, RangeInfo<Vec<NavigationTarget>>) {
    type Ctx = WorldSnapshot;
    type Output = req::GotoDefinitionResponse;
    fn try_conv_with(self, world: &WorldSnapshot) -> Result<req::GotoTypeDefinitionResponse> {
        let (file_id, RangeInfo { range, info: navs }) = self;
        let links = navs
            .into_iter()
            .map(|nav| (file_id, RangeInfo::new(range, nav)))
            .try_conv_with_to_vec(world)?;
        if world.options.supports_location_link {
            Ok(links.into())
        } else {
            let locations: Vec<Location> = links
                .into_iter()
                .map(|link| Location { uri: link.target_uri, range: link.target_selection_range })
                .collect();
            Ok(locations.into())
        }
    }
}

pub fn to_location(
    file_id: FileId,
    range: TextRange,
    world: &WorldSnapshot,
    line_index: &LineIndex,
) -> Result<Location> {
    let url = file_id.try_conv_with(world)?;
    let loc = Location::new(url, range.conv_with(line_index));
    Ok(loc)
}

pub trait MapConvWith<'a>: Sized + 'a {
    type Ctx;
    type Output;

    fn map_conv_with(self, ctx: &'a Self::Ctx) -> ConvWithIter<'a, Self, Self::Ctx> {
        ConvWithIter { iter: self, ctx }
    }
}

impl<'a, I> MapConvWith<'a> for I
where
    I: Iterator + 'a,
    I::Item: ConvWith,
{
    type Ctx = <I::Item as ConvWith>::Ctx;
    type Output = <I::Item as ConvWith>::Output;
}

pub struct ConvWithIter<'a, I, Ctx: 'a> {
    iter: I,
    ctx: &'a Ctx,
}

impl<'a, I, Ctx> Iterator for ConvWithIter<'a, I, Ctx>
where
    I: Iterator,
    I::Item: ConvWith<Ctx = Ctx>,
{
    type Item = <I::Item as ConvWith>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| item.conv_with(self.ctx))
    }
}

pub trait TryConvWithToVec<'a>: Sized + 'a {
    type Ctx;
    type Output;

    fn try_conv_with_to_vec(self, ctx: &'a Self::Ctx) -> Result<Vec<Self::Output>>;
}

impl<'a, I> TryConvWithToVec<'a> for I
where
    I: Iterator + 'a,
    I::Item: TryConvWith,
{
    type Ctx = <I::Item as TryConvWith>::Ctx;
    type Output = <I::Item as TryConvWith>::Output;

    fn try_conv_with_to_vec(self, ctx: &'a Self::Ctx) -> Result<Vec<Self::Output>> {
        self.map(|it| it.try_conv_with(ctx)).collect()
    }
}
