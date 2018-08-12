use languageserver_types::{Range, SymbolKind, Position};
use libeditor::{LineIndex, LineCol};
use libsyntax2::{SyntaxKind, TextUnit, TextRange};

pub trait Conv {
    type Output;
    fn conv(&self) -> Self::Output;
}

pub trait ConvWith {
    type Ctx;
    type Output;
    fn conv_with(&self, ctx: &Self::Ctx) -> Self::Output;
}

impl Conv for SyntaxKind {
    type Output = SymbolKind;

    fn conv(&self) -> <Self as Conv>::Output {
        match *self {
            SyntaxKind::FUNCTION => SymbolKind::Function,
            SyntaxKind::STRUCT => SymbolKind::Struct,
            SyntaxKind::ENUM => SymbolKind::Enum,
            SyntaxKind::TRAIT => SymbolKind::Interface,
            SyntaxKind::MODULE => SymbolKind::Module,
            SyntaxKind::TYPE_ITEM => SymbolKind::TypeParameter,
            SyntaxKind::STATIC_ITEM => SymbolKind::Constant,
            SyntaxKind::CONST_ITEM => SymbolKind::Constant,
            _ => SymbolKind::Variable,
        }
    }
}

impl ConvWith for Position {
    type Ctx = LineIndex;
    type Output = TextUnit;

    fn conv_with(&self, line_index: &LineIndex) -> TextUnit {
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

    fn conv_with(&self, line_index: &LineIndex) -> Position {
        let line_col = line_index.line_col(*self);
        // TODO: UTF-16
        Position::new(line_col.line as u64, u32::from(line_col.col) as u64)
    }
}

impl ConvWith for TextRange {
    type Ctx = LineIndex;
    type Output = Range;

    fn conv_with(&self, line_index: &LineIndex) -> Range {
        Range::new(
            self.start().conv_with(line_index),
            self.end().conv_with(line_index),
        )
    }
}

impl ConvWith for Range {
    type Ctx = LineIndex;
    type Output = TextRange;

    fn conv_with(&self, line_index: &LineIndex) -> TextRange {
        TextRange::from_to(
            self.start.conv_with(line_index),
            self.end.conv_with(line_index),
        )
    }
}
