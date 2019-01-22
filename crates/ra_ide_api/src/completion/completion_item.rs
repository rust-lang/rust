use hir::PerNs;

use crate::completion::completion_context::CompletionContext;
use ra_syntax::TextRange;
use ra_text_edit::TextEdit;

/// `CompletionItem` describes a single completion variant in the editor pop-up.
/// It is basically a POD with various properties. To construct a
/// `CompletionItem`, use `new` method and the `Builder` struct.
#[derive(Debug)]
pub struct CompletionItem {
    /// Used only internally in tests, to check only specific kind of
    /// completion.
    completion_kind: CompletionKind,
    label: String,
    kind: Option<CompletionItemKind>,
    detail: Option<String>,
    documentation: Option<String>,
    lookup: Option<String>,
    insert_text: Option<String>,
    insert_text_format: InsertTextFormat,
    /// Where completion occurs. `source_range` must contain the completion offset.
    /// `insert_text` should start with what `source_range` points to, or VSCode
    /// will filter out the completion silently.
    source_range: TextRange,
    /// Additional text edit, ranges in `text_edit` must never intersect with `source_range`.
    /// Or VSCode will drop it silently.
    text_edit: Option<TextEdit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionItemKind {
    Snippet,
    Keyword,
    Module,
    Function,
    Struct,
    Enum,
    EnumVariant,
    Binding,
    Field,
    Static,
    Const,
    Trait,
    TypeAlias,
    Method,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub(crate) enum CompletionKind {
    /// Parser-based keyword completion.
    Keyword,
    /// Your usual "complete all valid identifiers".
    Reference,
    /// "Secret sauce" completions.
    Magic,
    Snippet,
    Postfix,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum InsertTextFormat {
    PlainText,
    Snippet,
}

impl CompletionItem {
    pub(crate) fn new(
        completion_kind: CompletionKind,
        replace_range: TextRange,
        label: impl Into<String>,
    ) -> Builder {
        let label = label.into();
        Builder {
            source_range: replace_range,
            completion_kind,
            label,
            insert_text: None,
            insert_text_format: InsertTextFormat::PlainText,
            detail: None,
            documentation: None,
            lookup: None,
            kind: None,
            text_edit: None,
        }
    }
    /// What user sees in pop-up in the UI.
    pub fn label(&self) -> &str {
        &self.label
    }
    /// Short one-line additional information, like a type
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_ref().map(|it| it.as_str())
    }
    /// A doc-comment
    pub fn documentation(&self) -> Option<&str> {
        self.documentation.as_ref().map(|it| it.as_str())
    }
    /// What string is used for filtering.
    pub fn lookup(&self) -> &str {
        self.lookup
            .as_ref()
            .map(|it| it.as_str())
            .unwrap_or(self.label())
    }

    pub fn insert_text_format(&self) -> InsertTextFormat {
        self.insert_text_format.clone()
    }
    pub fn insert_text(&self) -> String {
        match &self.insert_text {
            Some(t) => t.clone(),
            None => self.label.clone(),
        }
    }
    pub fn kind(&self) -> Option<CompletionItemKind> {
        self.kind
    }
    pub fn take_text_edit(&mut self) -> Option<TextEdit> {
        self.text_edit.take()
    }
    pub fn source_range(&self) -> TextRange {
        self.source_range
    }
}

/// A helper to make `CompletionItem`s.
#[must_use]
pub(crate) struct Builder {
    source_range: TextRange,
    completion_kind: CompletionKind,
    label: String,
    insert_text: Option<String>,
    insert_text_format: InsertTextFormat,
    detail: Option<String>,
    documentation: Option<String>,
    lookup: Option<String>,
    kind: Option<CompletionItemKind>,
    text_edit: Option<TextEdit>,
}

impl Builder {
    pub(crate) fn add_to(self, acc: &mut Completions) {
        acc.add(self.build())
    }

    pub(crate) fn build(self) -> CompletionItem {
        CompletionItem {
            source_range: self.source_range,
            label: self.label,
            detail: self.detail,
            documentation: self.documentation,
            insert_text_format: self.insert_text_format,
            lookup: self.lookup,
            kind: self.kind,
            completion_kind: self.completion_kind,
            text_edit: self.text_edit,
            insert_text: self.insert_text,
        }
    }
    pub(crate) fn lookup_by(mut self, lookup: impl Into<String>) -> Builder {
        self.lookup = Some(lookup.into());
        self
    }
    pub(crate) fn insert_text(mut self, insert_text: impl Into<String>) -> Builder {
        self.insert_text = Some(insert_text.into());
        self
    }
    #[allow(unused)]
    pub(crate) fn insert_text_format(mut self, insert_text_format: InsertTextFormat) -> Builder {
        self.insert_text_format = insert_text_format;
        self
    }
    pub(crate) fn snippet(mut self, snippet: impl Into<String>) -> Builder {
        self.insert_text_format = InsertTextFormat::Snippet;
        self.insert_text(snippet)
    }
    pub(crate) fn kind(mut self, kind: CompletionItemKind) -> Builder {
        self.kind = Some(kind);
        self
    }
    #[allow(unused)]
    pub(crate) fn text_edit(mut self, edit: TextEdit) -> Builder {
        self.text_edit = Some(edit);
        self
    }
    #[allow(unused)]
    pub(crate) fn detail(self, detail: impl Into<String>) -> Builder {
        self.set_detail(Some(detail))
    }
    pub(crate) fn set_detail(mut self, detail: Option<impl Into<String>>) -> Builder {
        self.detail = detail.map(Into::into);
        self
    }
    #[allow(unused)]
    pub(crate) fn documentation(self, docs: impl Into<String>) -> Builder {
        self.set_documentation(Some(docs))
    }
    pub(crate) fn set_documentation(mut self, docs: Option<impl Into<String>>) -> Builder {
        self.documentation = docs.map(Into::into);
        self
    }
    pub(super) fn from_resolution(
        mut self,
        ctx: &CompletionContext,
        resolution: &hir::Resolution,
    ) -> Builder {
        let resolved = resolution.def_id.map(|d| d.resolve(ctx.db));
        let kind = match resolved {
            PerNs {
                types: Some(hir::Def::Module(..)),
                ..
            } => CompletionItemKind::Module,
            PerNs {
                types: Some(hir::Def::Struct(..)),
                ..
            } => CompletionItemKind::Struct,
            PerNs {
                types: Some(hir::Def::Enum(..)),
                ..
            } => CompletionItemKind::Enum,
            PerNs {
                types: Some(hir::Def::Trait(..)),
                ..
            } => CompletionItemKind::Trait,
            PerNs {
                types: Some(hir::Def::Type(..)),
                ..
            } => CompletionItemKind::TypeAlias,
            PerNs {
                values: Some(hir::Def::Const(..)),
                ..
            } => CompletionItemKind::Const,
            PerNs {
                values: Some(hir::Def::Static(..)),
                ..
            } => CompletionItemKind::Static,
            PerNs {
                values: Some(hir::Def::Function(function)),
                ..
            } => return self.from_function(ctx, function),
            _ => return self,
        };
        self.kind = Some(kind);
        self
    }

    pub(super) fn from_function(
        mut self,
        ctx: &CompletionContext,
        function: hir::Function,
    ) -> Builder {
        // If not an import, add parenthesis automatically.
        if ctx.use_item_syntax.is_none() && !ctx.is_call {
            if function.signature(ctx.db).params().is_empty() {
                self.insert_text = Some(format!("{}()$0", self.label));
            } else {
                self.insert_text = Some(format!("{}($0)", self.label));
            }
            self.insert_text_format = InsertTextFormat::Snippet;
        }
        if let Some(docs) = function.docs(ctx.db) {
            self.documentation = Some(docs);
        }

        if let Some(label) = function.label(ctx.db) {
            self.detail = Some(label);
        }

        self.kind = Some(CompletionItemKind::Function);
        self
    }
}

impl<'a> Into<CompletionItem> for Builder {
    fn into(self) -> CompletionItem {
        self.build()
    }
}

/// Represents an in-progress set of completions being built.
#[derive(Debug, Default)]
pub(crate) struct Completions {
    buf: Vec<CompletionItem>,
}

impl Completions {
    pub(crate) fn add(&mut self, item: impl Into<CompletionItem>) {
        self.buf.push(item.into())
    }
    pub(crate) fn add_all<I>(&mut self, items: I)
    where
        I: IntoIterator,
        I::Item: Into<CompletionItem>,
    {
        items.into_iter().for_each(|item| self.add(item.into()))
    }
}

impl Into<Vec<CompletionItem>> for Completions {
    fn into(self) -> Vec<CompletionItem> {
        self.buf
    }
}

#[cfg(test)]
pub(crate) fn check_completion(test_name: &str, code: &str, kind: CompletionKind) {
    use crate::mock_analysis::{single_file_with_position, analysis_and_position};
    use crate::completion::completions;
    use insta::assert_debug_snapshot_matches;
    let (analysis, position) = if code.contains("//-") {
        analysis_and_position(code)
    } else {
        single_file_with_position(code)
    };
    let completions = completions(&analysis.db, position).unwrap();
    let completion_items: Vec<CompletionItem> = completions.into();
    let kind_completions: Vec<CompletionItem> = completion_items
        .into_iter()
        .filter(|c| c.completion_kind == kind)
        .collect();
    assert_debug_snapshot_matches!(test_name, kind_completions);
}
