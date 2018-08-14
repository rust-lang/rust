use std::path::Path;

use languageserver_types::{Range, SymbolKind, Position, TextEdit, Location, Url};
use libeditor::{LineIndex, LineCol, Edit, AtomEdit};
use libsyntax2::{SyntaxKind, TextUnit, TextRange};

use Result;

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

impl<'a> TryConvWith for (&'a Path, TextRange) {
    type Ctx = LineIndex;
    type Output = Location;

    fn try_conv_with(self, line_index: &LineIndex) -> Result<Location> {
        let loc = Location::new(
            Url::from_file_path(self.0)
                .map_err(|()| format_err!("can't convert path to url: {}", self.0.display()))?,
            self.1.conv_with(line_index),
        );
        Ok(loc)
    }
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

