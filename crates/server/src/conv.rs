use languageserver_types::{
    Range, SymbolKind, Position, TextEdit, Location, Url,
    TextDocumentIdentifier, VersionedTextDocumentIdentifier, TextDocumentItem,
};
use libeditor::{LineIndex, LineCol, Edit, AtomEdit};
use libsyntax2::{SyntaxKind, TextUnit, TextRange};
use libanalysis::FileId;

use {Result, PathMap};

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

impl<'a> TryConvWith for &'a Url {
    type Ctx = PathMap;
    type Output = FileId;
    fn try_conv_with(self, path_map: &PathMap) -> Result<FileId> {
        let path = self.to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", self))?;
        path_map.get_id(&path).ok_or_else(|| format_err!("unknown file: {}", path.display()))
    }
}

impl TryConvWith for FileId {
    type Ctx = PathMap;
    type Output = Url;
    fn try_conv_with(self, path_map: &PathMap) -> Result<Url> {
        let path = path_map.get_path(self);
        let url = Url::from_file_path(path)
            .map_err(|()| format_err!("can't convert path to url: {}", path.display()))?;
        Ok(url)
    }
}

impl<'a> TryConvWith for &'a TextDocumentItem {
    type Ctx = PathMap;
    type Output = FileId;
    fn try_conv_with(self, path_map: &PathMap) -> Result<FileId> {
        self.uri.try_conv_with(path_map)
    }
}

impl<'a> TryConvWith for &'a VersionedTextDocumentIdentifier {
    type Ctx = PathMap;
    type Output = FileId;
    fn try_conv_with(self, path_map: &PathMap) -> Result<FileId> {
        self.uri.try_conv_with(path_map)
    }
}

impl<'a> TryConvWith for &'a TextDocumentIdentifier {
    type Ctx = PathMap;
    type Output = FileId;
    fn try_conv_with(self, path_map: &PathMap) -> Result<FileId> {
        self.uri.try_conv_with(path_map)
    }
}

pub fn to_location(
    file_id: FileId,
    range: TextRange,
    path_map: &PathMap,
    line_index: &LineIndex,
) -> Result<Location> {
        let url = file_id.try_conv_with(path_map)?;
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

