use hir::PerNs;
use ra_text_edit::{
    AtomTextEdit,
    TextEdit,
};

use crate::completion::completion_context::CompletionContext;

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
    lookup: Option<String>,
    /// The format of the insert text. The format applies to both the `insert_text` property
    /// and the `insert` property of a provided `text_edit`.
    insert_text_format: InsertTextFormat,
    /// An edit which is applied to a document when selecting this completion. When an edit is
    /// provided the value of `insert_text` is ignored.
    ///
    /// *Note:* The range of the edit must be a single line range and it must contain the position
    /// at which completion has been requested.
    ///
    /// *Note:* If sending a range that overlaps a string, the string should match the relevant
    /// part of the replacement text, or be filtered out.
    text_edit: Option<AtomTextEdit>,
    /// An optional array of additional text edits that are applied when
    /// selecting this completion. Edits must not overlap (including the same insert position)
    /// with the main edit nor with themselves.
    ///
    /// Additional text edits should be used to change text unrelated to the current cursor position
    /// (for example adding an import statement at the top of the file if the completion item will
    /// insert an unqualified type).
    additional_text_edits: Option<TextEdit>,
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
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum InsertTextFormat {
    PlainText,
    Snippet,
}

impl CompletionItem {
    pub(crate) fn new<'a>(
        completion_kind: CompletionKind,
        ctx: &'a CompletionContext,
        label: impl Into<String>,
    ) -> Builder<'a> {
        let label = label.into();
        Builder {
            ctx,
            completion_kind,
            label,
            insert_text: None,
            insert_text_format: InsertTextFormat::PlainText,
            detail: None,
            lookup: None,
            kind: None,
            text_edit: None,
            additional_text_edits: None,
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

    pub fn kind(&self) -> Option<CompletionItemKind> {
        self.kind
    }
    pub fn text_edit(&mut self) -> Option<&AtomTextEdit> {
        self.text_edit.as_ref()
    }
    pub fn take_additional_text_edits(&mut self) -> Option<TextEdit> {
        self.additional_text_edits.take()
    }
}

/// A helper to make `CompletionItem`s.
#[must_use]
pub(crate) struct Builder<'a> {
    ctx: &'a CompletionContext<'a>,
    completion_kind: CompletionKind,
    label: String,
    insert_text: Option<String>,
    insert_text_format: InsertTextFormat,
    detail: Option<String>,
    lookup: Option<String>,
    kind: Option<CompletionItemKind>,
    text_edit: Option<AtomTextEdit>,
    additional_text_edits: Option<TextEdit>,
}

impl<'a> Builder<'a> {
    pub(crate) fn add_to(self, acc: &mut Completions) {
        acc.add(self.build())
    }

    pub(crate) fn build(self) -> CompletionItem {
        let self_text_edit = self.text_edit;
        let self_insert_text = self.insert_text;
        let text_edit = match (self_text_edit, self_insert_text) {
            (Some(text_edit), ..) => Some(text_edit),
            (None, Some(insert_text)) => {
                Some(AtomTextEdit::replace(self.ctx.leaf_range(), insert_text))
            }
            _ => None,
        };

        CompletionItem {
            label: self.label,
            detail: self.detail,
            insert_text_format: self.insert_text_format,
            lookup: self.lookup,
            kind: self.kind,
            completion_kind: self.completion_kind,
            text_edit,
            additional_text_edits: self.additional_text_edits,
        }
    }
    pub(crate) fn lookup_by(mut self, lookup: impl Into<String>) -> Builder<'a> {
        self.lookup = Some(lookup.into());
        self
    }
    pub(crate) fn insert_text(mut self, insert_text: impl Into<String>) -> Builder<'a> {
        self.insert_text = Some(insert_text.into());
        self
    }
    pub(crate) fn insert_text_format(
        mut self,
        insert_text_format: InsertTextFormat,
    ) -> Builder<'a> {
        self.insert_text_format = insert_text_format;
        self
    }
    pub(crate) fn snippet(mut self, snippet: impl Into<String>) -> Builder<'a> {
        self.insert_text_format = InsertTextFormat::Snippet;
        self.insert_text(snippet)
    }
    pub(crate) fn kind(mut self, kind: CompletionItemKind) -> Builder<'a> {
        self.kind = Some(kind);
        self
    }
    pub(crate) fn text_edit(mut self, text_edit: AtomTextEdit) -> Builder<'a> {
        self.text_edit = Some(text_edit);
        self
    }
    pub(crate) fn additional_text_edits(mut self, additional_text_edits: TextEdit) -> Builder<'a> {
        self.additional_text_edits = Some(additional_text_edits);
        self
    }
    #[allow(unused)]
    pub(crate) fn detail(self, detail: impl Into<String>) -> Builder<'a> {
        self.set_detail(Some(detail))
    }
    pub(crate) fn set_detail(mut self, detail: Option<impl Into<String>>) -> Builder<'a> {
        self.detail = detail.map(Into::into);
        self
    }
    pub(super) fn from_resolution(
        mut self,
        ctx: &CompletionContext,
        resolution: &hir::Resolution,
    ) -> Builder<'a> {
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
    ) -> Builder<'a> {
        // If not an import, add parenthesis automatically.
        if ctx.use_item_syntax.is_none() && !ctx.is_call {
            if function.signature(ctx.db).params().is_empty() {
                self.insert_text = Some(format!("{}()$0", self.label));
            } else {
                self.insert_text = Some(format!("{}($0)", self.label));
            }
            self.insert_text_format = InsertTextFormat::Snippet;
        }
        self.kind = Some(CompletionItemKind::Function);
        self
    }
}

impl<'a> Into<CompletionItem> for Builder<'a> {
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
