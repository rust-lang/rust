//! Documentation attribute related utilities.
use either::Either;
use hir::{
    AttrId, AttrSourceMap, AttrsWithOwner, HasAttrs, InFile,
    db::{DefDatabase, HirDatabase},
    resolve_doc_path_on, sym,
};
use itertools::Itertools;
use span::{TextRange, TextSize};
use syntax::{
    AstToken,
    ast::{self, IsString},
};

/// Holds documentation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Documentation(String);

impl Documentation {
    pub fn new(s: String) -> Self {
        Documentation(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<Documentation> for String {
    fn from(Documentation(string): Documentation) -> Self {
        string
    }
}

pub trait HasDocs: HasAttrs {
    fn docs(self, db: &dyn HirDatabase) -> Option<Documentation>;
    fn docs_with_rangemap(self, db: &dyn HirDatabase) -> Option<(Documentation, DocsRangeMap)>;
    fn resolve_doc_path(
        self,
        db: &dyn HirDatabase,
        link: &str,
        ns: Option<hir::Namespace>,
        is_inner_doc: bool,
    ) -> Option<hir::DocLinkDef>;
}
/// A struct to map text ranges from [`Documentation`] back to TextRanges in the syntax tree.
#[derive(Debug)]
pub struct DocsRangeMap {
    source_map: AttrSourceMap,
    // (docstring-line-range, attr_index, attr-string-range)
    // a mapping from the text range of a line of the [`Documentation`] to the attribute index and
    // the original (untrimmed) syntax doc line
    mapping: Vec<(TextRange, AttrId, TextRange)>,
}

impl DocsRangeMap {
    /// Maps a [`TextRange`] relative to the documentation string back to its AST range
    pub fn map(&self, range: TextRange) -> Option<(InFile<TextRange>, AttrId)> {
        let found = self.mapping.binary_search_by(|(probe, ..)| probe.ordering(range)).ok()?;
        let (line_docs_range, idx, original_line_src_range) = self.mapping[found];
        if !line_docs_range.contains_range(range) {
            return None;
        }

        let relative_range = range - line_docs_range.start();

        let InFile { file_id, value: source } = self.source_map.source_of_id(idx);
        match source {
            Either::Left(attr) => {
                let string = get_doc_string_in_attr(attr)?;
                let text_range = string.open_quote_text_range()?;
                let range = TextRange::at(
                    text_range.end() + original_line_src_range.start() + relative_range.start(),
                    string.syntax().text_range().len().min(range.len()),
                );
                Some((InFile { file_id, value: range }, idx))
            }
            Either::Right(comment) => {
                let text_range = comment.syntax().text_range();
                let range = TextRange::at(
                    text_range.start()
                        + TextSize::try_from(comment.prefix().len()).ok()?
                        + original_line_src_range.start()
                        + relative_range.start(),
                    text_range.len().min(range.len()),
                );
                Some((InFile { file_id, value: range }, idx))
            }
        }
    }

    pub fn shift_docstring_line_range(self, offset: TextSize) -> DocsRangeMap {
        let mapping = self
            .mapping
            .into_iter()
            .map(|(buf_offset, id, base_offset)| {
                let buf_offset = buf_offset.checked_add(offset).unwrap();
                (buf_offset, id, base_offset)
            })
            .collect_vec();
        DocsRangeMap { source_map: self.source_map, mapping }
    }
}

pub fn docs_with_rangemap(
    db: &dyn DefDatabase,
    attrs: &AttrsWithOwner,
) -> Option<(Documentation, DocsRangeMap)> {
    let docs = attrs
        .by_key(sym::doc)
        .attrs()
        .filter_map(|attr| attr.string_value_unescape().map(|s| (s, attr.id)));
    let indent = doc_indent(attrs);
    let mut buf = String::new();
    let mut mapping = Vec::new();
    for (doc, idx) in docs {
        if !doc.is_empty() {
            let mut base_offset = 0;
            for raw_line in doc.split('\n') {
                let line = raw_line.trim_end();
                let line_len = line.len();
                let (offset, line) = match line.char_indices().nth(indent) {
                    Some((offset, _)) => (offset, &line[offset..]),
                    None => (0, line),
                };
                let buf_offset = buf.len();
                buf.push_str(line);
                mapping.push((
                    TextRange::new(buf_offset.try_into().ok()?, buf.len().try_into().ok()?),
                    idx,
                    TextRange::at(
                        (base_offset + offset).try_into().ok()?,
                        line_len.try_into().ok()?,
                    ),
                ));
                buf.push('\n');
                base_offset += raw_line.len() + 1;
            }
        } else {
            buf.push('\n');
        }
    }
    buf.pop();
    if buf.is_empty() {
        None
    } else {
        Some((Documentation(buf), DocsRangeMap { mapping, source_map: attrs.source_map(db) }))
    }
}

pub fn docs_from_attrs(attrs: &hir::Attrs) -> Option<String> {
    let docs = attrs.by_key(sym::doc).attrs().filter_map(|attr| attr.string_value_unescape());
    let indent = doc_indent(attrs);
    let mut buf = String::new();
    for doc in docs {
        // str::lines doesn't yield anything for the empty string
        if !doc.is_empty() {
            // We don't trim trailing whitespace from doc comments as multiple trailing spaces
            // indicates a hard line break in Markdown.
            let lines = doc.lines().map(|line| {
                line.char_indices().nth(indent).map_or(line, |(offset, _)| &line[offset..])
            });

            buf.extend(Itertools::intersperse(lines, "\n"));
        }
        buf.push('\n');
    }
    buf.pop();
    if buf.is_empty() { None } else { Some(buf) }
}

macro_rules! impl_has_docs {
    ($($def:ident,)*) => {$(
        impl HasDocs for hir::$def {
            fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
                docs_from_attrs(&self.attrs(db)).map(Documentation)
            }
            fn docs_with_rangemap(
                self,
                db: &dyn HirDatabase,
            ) -> Option<(Documentation, DocsRangeMap)> {
                docs_with_rangemap(db, &self.attrs(db))
            }
            fn resolve_doc_path(
                self,
                db: &dyn HirDatabase,
                link: &str,
                ns: Option<hir::Namespace>,
                is_inner_doc: bool,
            ) -> Option<hir::DocLinkDef> {
                resolve_doc_path_on(db, self, link, ns, is_inner_doc)
            }
        }
    )*};
}

impl_has_docs![
    Variant, Field, Static, Const, Trait, TraitAlias, TypeAlias, Macro, Function, Adt, Module,
    Impl, Crate,
];

macro_rules! impl_has_docs_enum {
    ($($variant:ident),* for $enum:ident) => {$(
        impl HasDocs for hir::$variant {
            fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
                hir::$enum::$variant(self).docs(db)
            }

            fn docs_with_rangemap(
                self,
                db: &dyn HirDatabase,
            ) -> Option<(Documentation, DocsRangeMap)> {
                hir::$enum::$variant(self).docs_with_rangemap(db)
            }
            fn resolve_doc_path(
                self,
                db: &dyn HirDatabase,
                link: &str,
                ns: Option<hir::Namespace>,
                is_inner_doc: bool,
            ) -> Option<hir::DocLinkDef> {
                hir::$enum::$variant(self).resolve_doc_path(db, link, ns, is_inner_doc)
            }
        }
    )*};
}

impl_has_docs_enum![Struct, Union, Enum for Adt];

impl HasDocs for hir::AssocItem {
    fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
        match self {
            hir::AssocItem::Function(it) => it.docs(db),
            hir::AssocItem::Const(it) => it.docs(db),
            hir::AssocItem::TypeAlias(it) => it.docs(db),
        }
    }

    fn docs_with_rangemap(self, db: &dyn HirDatabase) -> Option<(Documentation, DocsRangeMap)> {
        match self {
            hir::AssocItem::Function(it) => it.docs_with_rangemap(db),
            hir::AssocItem::Const(it) => it.docs_with_rangemap(db),
            hir::AssocItem::TypeAlias(it) => it.docs_with_rangemap(db),
        }
    }

    fn resolve_doc_path(
        self,
        db: &dyn HirDatabase,
        link: &str,
        ns: Option<hir::Namespace>,
        is_inner_doc: bool,
    ) -> Option<hir::DocLinkDef> {
        match self {
            hir::AssocItem::Function(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
            hir::AssocItem::Const(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
            hir::AssocItem::TypeAlias(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        }
    }
}

impl HasDocs for hir::ExternCrateDecl {
    fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
        let crate_docs = docs_from_attrs(&self.resolved_crate(db)?.root_module().attrs(db));
        let decl_docs = docs_from_attrs(&self.attrs(db));
        match (decl_docs, crate_docs) {
            (None, None) => None,
            (Some(decl_docs), None) => Some(decl_docs),
            (None, Some(crate_docs)) => Some(crate_docs),
            (Some(mut decl_docs), Some(crate_docs)) => {
                decl_docs.push('\n');
                decl_docs.push('\n');
                decl_docs += &crate_docs;
                Some(decl_docs)
            }
        }
        .map(Documentation::new)
    }

    fn docs_with_rangemap(self, db: &dyn HirDatabase) -> Option<(Documentation, DocsRangeMap)> {
        let crate_docs = docs_with_rangemap(db, &self.resolved_crate(db)?.root_module().attrs(db));
        let decl_docs = docs_with_rangemap(db, &self.attrs(db));
        match (decl_docs, crate_docs) {
            (None, None) => None,
            (Some(decl_docs), None) => Some(decl_docs),
            (None, Some(crate_docs)) => Some(crate_docs),
            (
                Some((Documentation(mut decl_docs), mut decl_range_map)),
                Some((Documentation(crate_docs), crate_range_map)),
            ) => {
                decl_docs.push('\n');
                decl_docs.push('\n');
                let offset = TextSize::new(decl_docs.len() as u32);
                decl_docs += &crate_docs;
                let crate_range_map = crate_range_map.shift_docstring_line_range(offset);
                decl_range_map.mapping.extend(crate_range_map.mapping);
                Some((Documentation(decl_docs), decl_range_map))
            }
        }
    }
    fn resolve_doc_path(
        self,
        db: &dyn HirDatabase,
        link: &str,
        ns: Option<hir::Namespace>,
        is_inner_doc: bool,
    ) -> Option<hir::DocLinkDef> {
        resolve_doc_path_on(db, self, link, ns, is_inner_doc)
    }
}

fn get_doc_string_in_attr(it: &ast::Attr) -> Option<ast::String> {
    match it.expr() {
        // #[doc = lit]
        Some(ast::Expr::Literal(lit)) => match lit.kind() {
            ast::LiteralKind::String(it) => Some(it),
            _ => None,
        },
        // #[cfg_attr(..., doc = "", ...)]
        None => {
            // FIXME: See highlight injection for what to do here
            None
        }
        _ => None,
    }
}

fn doc_indent(attrs: &hir::Attrs) -> usize {
    let mut min = !0;
    for val in attrs.by_key(sym::doc).attrs().filter_map(|attr| attr.string_value_unescape()) {
        if let Some(m) =
            val.lines().filter_map(|line| line.chars().position(|c| !c.is_whitespace())).min()
        {
            min = min.min(m);
        }
    }
    min
}
