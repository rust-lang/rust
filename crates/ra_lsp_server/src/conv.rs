use languageserver_types::{
    self, Location, Position, Range, SymbolKind, TextDocumentEdit, TextDocumentIdentifier,
    TextDocumentItem, TextDocumentPositionParams, Url, VersionedTextDocumentIdentifier, InsertTextFormat,
};
use ra_analysis::{FileId, FileSystemEdit, SourceChange, SourceFileEdit, FilePosition, CompletionItem, CompletionItemKind, InsertText};
use ra_editor::{LineCol, LineIndex};
use ra_text_edit::{AtomTextEdit, TextEdit};
use ra_syntax::{SyntaxKind, TextRange, TextUnit};

use crate::{req, server_world::ServerWorld, Result};

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

impl Conv for CompletionItemKind {
    type Output = ::languageserver_types::CompletionItemKind;

    fn conv(self) -> <Self as Conv>::Output {
        use ::languageserver_types::CompletionItemKind::*;
        match self {
            CompletionItemKind::Keyword => Keyword,
            CompletionItemKind::Snippet => Snippet,
            CompletionItemKind::Module => Module,
            CompletionItemKind::Function => Function,
            CompletionItemKind::Binding => Variable,
        }
    }
}

impl Conv for CompletionItem {
    type Output = ::languageserver_types::CompletionItem;

    fn conv(self) -> <Self as Conv>::Output {
        let mut res = ::languageserver_types::CompletionItem {
            label: self.label().to_string(),
            filter_text: Some(self.lookup().to_string()),
            kind: self.kind().map(|it| it.conv()),
            ..Default::default()
        };
        match self.insert_text() {
            InsertText::PlainText { text } => {
                res.insert_text = Some(text);
                res.insert_text_format = Some(InsertTextFormat::PlainText);
            }
            InsertText::Snippet { text } => {
                res.insert_text = Some(text);
                res.insert_text_format = Some(InsertTextFormat::Snippet);
                res.kind = Some(languageserver_types::CompletionItemKind::Keyword);
            }
        }
        res
    }
}

impl ConvWith for Position {
    type Ctx = LineIndex;
    type Output = TextUnit;

    fn conv_with(self, line_index: &LineIndex) -> TextUnit {
        let line_col = LineCol {
            line: self.line as u32,
            col_utf16: self.character as u32,
        };
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

impl ConvWith for TextEdit {
    type Ctx = LineIndex;
    type Output = Vec<languageserver_types::TextEdit>;

    fn conv_with(self, line_index: &LineIndex) -> Vec<languageserver_types::TextEdit> {
        self.as_atoms()
            .into_iter()
            .map_conv_with(line_index)
            .collect()
    }
}

impl<'a> ConvWith for &'a AtomTextEdit {
    type Ctx = LineIndex;
    type Output = languageserver_types::TextEdit;

    fn conv_with(self, line_index: &LineIndex) -> languageserver_types::TextEdit {
        languageserver_types::TextEdit {
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

impl<'a> TryConvWith for &'a TextDocumentPositionParams {
    type Ctx = ServerWorld;
    type Output = FilePosition;
    fn try_conv_with(self, world: &ServerWorld) -> Result<FilePosition> {
        let file_id = self.text_document.try_conv_with(world)?;
        let line_index = world.analysis().file_line_index(file_id);
        let offset = self.position.conv_with(&line_index);
        Ok(FilePosition { file_id, offset })
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
                let edits = self
                    .source_file_edits
                    .iter()
                    .find(|it| it.file_id == pos.file_id)
                    .map(|it| it.edit.as_atoms())
                    .unwrap_or(&[]);
                let line_col = translate_offset_with_edit(&*line_index, pos.offset, edits);
                let position =
                    Position::new(u64::from(line_col.line), u64::from(line_col.col_utf16));
                Some(TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier::new(pos.file_id.try_conv_with(world)?),
                    position,
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

// HACK: we should translate offset to line/column using linde_index *with edits applied*.
// A naive version of this function would be to apply `edits` to the original text,
// construct a new line index and use that, but it would be slow.
//
// Writing fast & correct version is issue #105, let's use a quick hack in the meantime
fn translate_offset_with_edit(
    pre_edit_index: &LineIndex,
    offset: TextUnit,
    edits: &[AtomTextEdit],
) -> LineCol {
    let fallback = pre_edit_index.line_col(offset);
    let edit = match edits.first() {
        None => return fallback,
        Some(edit) => edit,
    };
    let end_offset = edit.delete.start() + TextUnit::of_str(&edit.insert);
    if !(edit.delete.start() <= offset && offset <= end_offset) {
        return fallback;
    }
    let rel_offset = offset - edit.delete.start();
    let in_edit_line_col = LineIndex::new(&edit.insert).line_col(rel_offset);
    let edit_line_col = pre_edit_index.line_col(edit.delete.start());
    if in_edit_line_col.line == 0 {
        LineCol {
            line: edit_line_col.line,
            col_utf16: edit_line_col.col_utf16 + in_edit_line_col.col_utf16,
        }
    } else {
        LineCol {
            line: edit_line_col.line + in_edit_line_col.line,
            col_utf16: in_edit_line_col.col_utf16,
        }
    }
}

#[derive(Debug)]
struct OffsetNewlineIter<'a> {
    text: &'a str,
    offset: TextUnit,
}

impl<'a> Iterator for OffsetNewlineIter<'a> {
    type Item = TextUnit;
    fn next(&mut self) -> Option<TextUnit> {
        let next_idx = self
            .text
            .char_indices()
            .filter_map(|(i, c)| if c == '\n' { Some(i + 1) } else { None })
            .next()?;
        let next = self.offset + TextUnit::from_usize(next_idx);
        self.text = &self.text[next_idx..];
        self.offset = next;
        Some(next)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TranslatedPos {
    Before,
    After,
}

/// None means it was deleted
type TranslatedOffset = Option<(TranslatedPos, TextUnit)>;

fn translate_offset(offset: TextUnit, edit: &TranslatedAtomEdit) -> TranslatedOffset {
    if offset <= edit.delete.start() {
        Some((TranslatedPos::Before, offset))
    } else if offset <= edit.delete.end() {
        None
    } else {
        let diff = edit.insert.len() as i64 - edit.delete.len().to_usize() as i64;
        let after = TextUnit::from((offset.to_usize() as i64 + diff) as u32);
        Some((TranslatedPos::After, after))
    }
}

trait TranslatedNewlineIterator {
    fn translate(&self, offset: TextUnit) -> TextUnit;
    fn translate_range(&self, range: TextRange) -> TextRange {
        TextRange::from_to(self.translate(range.start()), self.translate(range.end()))
    }
    fn next_translated(&mut self) -> Option<TextUnit>;
    fn boxed<'a>(self) -> Box<TranslatedNewlineIterator + 'a>
    where
        Self: 'a + Sized,
    {
        Box::new(self)
    }
}

struct TranslatedAtomEdit<'a> {
    delete: TextRange,
    insert: &'a str,
}

struct TranslatedNewlines<'a, T: TranslatedNewlineIterator> {
    inner: T,
    next_inner: Option<TranslatedOffset>,
    edit: TranslatedAtomEdit<'a>,
    insert: OffsetNewlineIter<'a>,
}

impl<'a, T: TranslatedNewlineIterator> TranslatedNewlines<'a, T> {
    fn from(inner: T, edit: &'a AtomTextEdit) -> Self {
        let delete = inner.translate_range(edit.delete);
        let mut res = TranslatedNewlines {
            inner,
            next_inner: None,
            edit: TranslatedAtomEdit {
                delete,
                insert: &edit.insert,
            },
            insert: OffsetNewlineIter {
                offset: delete.start(),
                text: &edit.insert,
            },
        };
        // prepare next_inner
        res.advance_inner();
        res
    }

    fn advance_inner(&mut self) {
        self.next_inner = self
            .inner
            .next_translated()
            .map(|x| translate_offset(x, &self.edit));
    }
}

impl<'a, T: TranslatedNewlineIterator> TranslatedNewlineIterator for TranslatedNewlines<'a, T> {
    fn translate(&self, offset: TextUnit) -> TextUnit {
        let offset = self.inner.translate(offset);
        let (_, offset) =
            translate_offset(offset, &self.edit).expect("translate_unit returned None");
        offset
    }

    fn next_translated(&mut self) -> Option<TextUnit> {
        match self.next_inner {
            None => self.insert.next(),
            Some(next) => match next {
                None => self.insert.next().or_else(|| {
                    self.advance_inner();
                    self.next_translated()
                }),
                Some((TranslatedPos::Before, next)) => {
                    self.advance_inner();
                    Some(next)
                }
                Some((TranslatedPos::After, next)) => self.insert.next().or_else(|| {
                    self.advance_inner();
                    Some(next)
                }),
            },
        }
    }
}

impl<'a> Iterator for Box<dyn TranslatedNewlineIterator + 'a> {
    type Item = TextUnit;
    fn next(&mut self) -> Option<TextUnit> {
        self.next_translated()
    }
}

impl<T: TranslatedNewlineIterator + ?Sized> TranslatedNewlineIterator for Box<T> {
    fn translate(&self, offset: TextUnit) -> TextUnit {
        self.as_ref().translate(offset)
    }
    fn next_translated(&mut self) -> Option<TextUnit> {
        self.as_mut().next_translated()
    }
}

struct IteratorWrapper<T: Iterator<Item = TextUnit>>(T);

impl<T: Iterator<Item = TextUnit>> TranslatedNewlineIterator for IteratorWrapper<T> {
    fn translate(&self, offset: TextUnit) -> TextUnit {
        offset
    }
    fn next_translated(&mut self) -> Option<TextUnit> {
        self.0.next()
    }
}

impl<T: Iterator<Item = TextUnit>> Iterator for IteratorWrapper<T> {
    type Item = TextUnit;
    fn next(&mut self) -> Option<TextUnit> {
        self.0.next()
    }
}

fn translate_newlines<'a>(
    mut newlines: Box<TranslatedNewlineIterator + 'a>,
    edits: &'a [AtomTextEdit],
) -> Box<TranslatedNewlineIterator + 'a> {
    for edit in edits {
        newlines = TranslatedNewlines::from(newlines, edit).boxed();
    }
    newlines
}

#[allow(dead_code)]
fn translate_offset_with_edit_fast(
    pre_edit_index: &LineIndex,
    offset: TextUnit,
    edits: &[AtomTextEdit],
) -> LineCol {
    // println!("{:?}", pre_edit_index.newlines());
    let mut newlines: Box<TranslatedNewlineIterator> = Box::new(IteratorWrapper(
        pre_edit_index.newlines().iter().map(|x| *x),
    ));

    newlines = translate_newlines(newlines, edits);

    let mut line = 0;
    for n in newlines {
        if n > offset {
            break;
        }
        line += 1;
    }

    LineCol {
        line: line,
        col_utf16: 0,
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
        let edits = self
            .edit
            .as_atoms()
            .iter()
            .map_conv_with(&line_index)
            .collect();
        Ok(TextDocumentEdit {
            text_document,
            edits,
        })
    }
}

impl TryConvWith for FileSystemEdit {
    type Ctx = ServerWorld;
    type Output = req::FileSystemEdit;
    fn try_conv_with(self, world: &ServerWorld) -> Result<req::FileSystemEdit> {
        let res = match self {
            FileSystemEdit::CreateFile { source_root, path } => {
                let uri = world.path_to_uri(source_root, &path)?;
                req::FileSystemEdit::CreateFile { uri }
            }
            FileSystemEdit::MoveFile {
                src,
                dst_source_root,
                dst_path,
            } => {
                let src = world.file_id_to_uri(src)?;
                let dst = world.path_to_uri(dst_source_root, &dst_path)?;
                req::FileSystemEdit::MoveFile { src, dst }
            }
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

#[cfg(test)]
mod test {
    use proptest::{prelude::*, proptest, proptest_helper};
    use super::*;
    use ra_text_edit::test_utils::{arb_text, arb_offset, arb_edits};

    #[derive(Debug)]
    struct ArbTextWithOffsetAndEdits {
        text: String,
        offset: TextUnit,
        edits: Vec<AtomTextEdit>,
    }

    fn arb_text_with_offset_and_edits() -> BoxedStrategy<ArbTextWithOffsetAndEdits> {
        arb_text()
            .prop_flat_map(|text| {
                (arb_offset(&text), arb_edits(&text), Just(text)).prop_map(
                    |(offset, edits, text)| ArbTextWithOffsetAndEdits {
                        text,
                        offset,
                        edits,
                    },
                )
            })
            .boxed()
    }

    fn edit_text(pre_edit_text: &str, mut edits: Vec<AtomTextEdit>) -> String {
        // apply edits ordered from last to first
        // since they should not overlap we can just use start()
        edits.sort_by_key(|x| -(x.delete.start().to_usize() as isize));

        let mut text = pre_edit_text.to_owned();

        for edit in &edits {
            let range = edit.delete.start().to_usize()..edit.delete.end().to_usize();
            text.replace_range(range, &edit.insert);
        }

        text
    }

    fn translate_after_edit(
        pre_edit_text: &str,
        offset: TextUnit,
        edits: Vec<AtomTextEdit>,
    ) -> LineCol {
        let text = edit_text(pre_edit_text, edits);
        let line_index = LineIndex::new(&text);
        line_index.line_col(offset)
    }

    proptest! {
        #[test]
        fn test_translate_offset_with_edit(x in arb_text_with_offset_and_edits()) {
            let line_index = LineIndex::new(&x.text);
            let expected = translate_after_edit(&x.text, x.offset, x.edits.clone());
            let actual = translate_offset_with_edit_fast(&line_index, x.offset, &x.edits);
            assert_eq!(actual.line, expected.line);
        }
    }

    #[test]
    fn test_translate_offset_with_edit_1() {
        let x = ArbTextWithOffsetAndEdits {
            text: "jbnan".to_owned(),
            offset: 3.into(),
            edits: vec![
                AtomTextEdit::delete(TextRange::from_to(1.into(), 3.into())),
                AtomTextEdit::insert(4.into(), "\n".into()),
            ],
        };
        let line_index = LineIndex::new(&x.text);
        let expected = translate_after_edit(&x.text, x.offset, x.edits.clone());
        let actual = translate_offset_with_edit_fast(&line_index, x.offset, &x.edits);
        // assert_eq!(actual, expected);
        assert_eq!(actual.line, expected.line);
    }

    #[test]
    fn test_translate_offset_with_edit_2() {
        let x = ArbTextWithOffsetAndEdits {
            text: "aa\n".to_owned(),
            offset: 1.into(),
            edits: vec![AtomTextEdit::delete(TextRange::from_to(0.into(), 2.into()))],
        };
        let line_index = LineIndex::new(&x.text);
        let expected = translate_after_edit(&x.text, x.offset, x.edits.clone());
        let actual = translate_offset_with_edit_fast(&line_index, x.offset, &x.edits);
        // assert_eq!(actual, expected);
        assert_eq!(actual.line, expected.line);
    }

    #[test]
    fn test_translate_offset_with_edit_3() {
        let x = ArbTextWithOffsetAndEdits {
            text: "".to_owned(),
            offset: 0.into(),
            edits: vec![AtomTextEdit::insert(0.into(), "\n".into())],
        };
        let line_index = LineIndex::new(&x.text);
        let expected = translate_after_edit(&x.text, x.offset, x.edits.clone());
        let actual = translate_offset_with_edit_fast(&line_index, x.offset, &x.edits);
        // assert_eq!(actual, expected);
        assert_eq!(actual.line, expected.line);
    }

}
