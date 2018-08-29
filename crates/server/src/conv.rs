use languageserver_types::{
    Range, SymbolKind, Position, TextEdit, Location, Url,
    TextDocumentIdentifier, VersionedTextDocumentIdentifier, TextDocumentItem,
    TextDocumentPositionParams, TextDocumentEdit,
};
use libeditor::{LineIndex, LineCol, Edit, AtomEdit};
use libsyntax2::{SyntaxKind, TextUnit, TextRange};
use libanalysis::{FileId, SourceChange, SourceFileEdit, FileSystemEdit};

use {
    Result,
    server_world::ServerWorld,
    req,
};

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
            SyntaxKind::TRAIT_DEF => SymbolKind::Interface,
            SyntaxKind::MODULE => SymbolKind::Module,
            SyntaxKind::TYPE_DEF => SymbolKind::TypeParameter,
            SyntaxKind::STATIC_DEF => SymbolKind::Constant,
            SyntaxKind::CONST_DEF => SymbolKind::Constant,
            SyntaxKind::IMPL_ITEM => SymbolKind::Object,
            _ => SymbolKind::Variable,
        }
    }
}

impl ConvWith for Position {
    type Ctx = LineIndex;
    type Output = TextUnit;

    fn conv_with(self, line_index: &LineIndex) -> TextUnit {
        // TODO: UTF-16
        let line_col = LineCol {
            line: self.line as u32,
            col: (self.character as u32).into(),
        };
        line_index.offset(line_col)
    }
}

impl ConvWith for TextUnit {
    type Ctx = LineIndex;
    type Output = Position;

    fn conv_with(self, line_index: &LineIndex) -> Position {
        let line_col = line_index.line_col(self);
        // TODO: UTF-16
        Position::new(line_col.line as u64, u32::from(line_col.col) as u64)
    }
}

impl ConvWith for TextRange {
    type Ctx = LineIndex;
    type Output = Range;

    fn conv_with(self, line_index: &LineIndex) -> Range {
        Range::new(
            self.start().conv_with(line_index),
            self.end().conv_with(line_index),
        )
    }
}

impl ConvWith for Range {
    type Ctx = LineIndex;
    type Output = TextRange;

    fn conv_with(self, line_index: &LineIndex) -> TextRange {
        TextRange::from_to(
            self.start.conv_with(line_index),
            self.end.conv_with(line_index),
        )
    }
}

impl ConvWith for Edit {
    type Ctx = LineIndex;
    type Output = Vec<TextEdit>;

    fn conv_with(self, line_index: &LineIndex) -> Vec<TextEdit> {
        self.into_atoms()
            .into_iter()
            .map_conv_with(line_index)
            .collect()
    }
}

impl ConvWith for AtomEdit {
    type Ctx = LineIndex;
    type Output = TextEdit;

    fn conv_with(self, line_index: &LineIndex) -> TextEdit {
        TextEdit {
            range: self.delete.conv_with(line_index),
            new_text: self.insert,
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
    type Ctx = ServerWorld;
    type Output = FileId;
    fn try_conv_with(self, world: &ServerWorld) -> Result<FileId> {
        world.uri_to_file_id(self)
    }
}

impl TryConvWith for FileId {
    type Ctx = ServerWorld;
    type Output = Url;
    fn try_conv_with(self, world: &ServerWorld) -> Result<Url> {
        world.file_id_to_uri(self)
    }
}

impl<'a> TryConvWith for &'a TextDocumentItem {
    type Ctx = ServerWorld;
    type Output = FileId;
    fn try_conv_with(self, world: &ServerWorld) -> Result<FileId> {
        self.uri.try_conv_with(world)
    }
}

impl<'a> TryConvWith for &'a VersionedTextDocumentIdentifier {
    type Ctx = ServerWorld;
    type Output = FileId;
    fn try_conv_with(self, world: &ServerWorld) -> Result<FileId> {
        self.uri.try_conv_with(world)
    }
}

impl<'a> TryConvWith for &'a TextDocumentIdentifier {
    type Ctx = ServerWorld;
    type Output = FileId;
    fn try_conv_with(self, world: &ServerWorld) -> Result<FileId> {
        world.uri_to_file_id(&self.uri)
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
    type Ctx = ServerWorld;
    type Output = req::SourceChange;
    fn try_conv_with(self, world: &ServerWorld) -> Result<req::SourceChange> {
        let cursor_position = match self.cursor_position {
            None => None,
            Some(pos) => {
                let line_index = world.analysis().file_line_index(pos.file_id);
                Some(TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier::new(pos.file_id.try_conv_with(world)?),
                    position: pos.offset.conv_with(&line_index),
                })
            }
        };
        let source_file_edits = self.source_file_edits.try_conv_with(world)?;
        let file_system_edits = self.file_system_edits.try_conv_with(world)?;
        Ok(req::SourceChange {
            label: self.label,
            source_file_edits,
            file_system_edits,
            cursor_position,
        })
    }
}

impl TryConvWith for SourceFileEdit {
    type Ctx = ServerWorld;
    type Output = TextDocumentEdit;
    fn try_conv_with(self, world: &ServerWorld) -> Result<TextDocumentEdit> {
        let text_document = VersionedTextDocumentIdentifier {
            uri: self.file_id.try_conv_with(world)?,
            version: None,
        };
        let line_index = world.analysis().file_line_index(self.file_id);
        let edits = self.edits
            .into_iter()
            .map_conv_with(&line_index)
            .collect();
        Ok(TextDocumentEdit { text_document, edits })
    }
}

impl TryConvWith for FileSystemEdit {
    type Ctx = ServerWorld;
    type Output = req::FileSystemEdit;
    fn try_conv_with(self, world: &ServerWorld) -> Result<req::FileSystemEdit> {
        let res = match self {
            FileSystemEdit::CreateFile { anchor, path } => {
                let uri = world.file_id_to_uri(anchor)?;
                let path = &path.as_str()[3..]; // strip `../` b/c url is weird
                let uri = uri.join(path)?;
                req::FileSystemEdit::CreateFile { uri }
            },
            FileSystemEdit::MoveFile { file, path } => {
                let src = world.file_id_to_uri(file)?;
                let path = &path.as_str()[3..]; // strip `../` b/c url is weird
                let dst = src.join(path)?;
                req::FileSystemEdit::MoveFile { src, dst }
            },
        };
        Ok(res)
    }
}

pub fn to_location(
    file_id: FileId,
    range: TextRange,
    world: &ServerWorld,
    line_index: &LineIndex,
) -> Result<Location> {
        let url = file_id.try_conv_with(world)?;
        let loc = Location::new(
            url,
            range.conv_with(line_index),
        );
        Ok(loc)
}

pub trait MapConvWith<'a>: Sized {
    type Ctx;
    type Output;

    fn map_conv_with(self, ctx: &'a Self::Ctx) -> ConvWithIter<'a, Self, Self::Ctx> {
        ConvWithIter { iter: self, ctx }
    }
}

impl<'a, I> MapConvWith<'a> for I
    where I: Iterator,
          I::Item: ConvWith
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
        I::Item: ConvWith<Ctx=Ctx>,
{
    type Item = <I::Item as ConvWith>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| item.conv_with(self.ctx))
    }
}

