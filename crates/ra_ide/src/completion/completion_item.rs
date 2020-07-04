//! FIXME: write short doc here

use std::fmt;

use hir::Documentation;
use ra_syntax::TextRange;
use ra_text_edit::TextEdit;

use crate::completion::completion_config::SnippetCap;

/// `CompletionItem` describes a single completion variant in the editor pop-up.
/// It is basically a POD with various properties. To construct a
/// `CompletionItem`, use `new` method and the `Builder` struct.
pub struct CompletionItem {
    /// Used only internally in tests, to check only specific kind of
    /// completion (postfix, keyword, reference, etc).
    #[allow(unused)]
    pub(crate) completion_kind: CompletionKind,
    /// Label in the completion pop up which identifies completion.
    label: String,
    /// Range of identifier that is being completed.
    ///
    /// It should be used primarily for UI, but we also use this to convert
    /// genetic TextEdit into LSP's completion edit (see conv.rs).
    ///
    /// `source_range` must contain the completion offset. `insert_text` should
    /// start with what `source_range` points to, or VSCode will filter out the
    /// completion silently.
    source_range: TextRange,
    /// What happens when user selects this item.
    ///
    /// Typically, replaces `source_range` with new identifier.
    text_edit: TextEdit,
    insert_text_format: InsertTextFormat,

    /// What item (struct, function, etc) are we completing.
    kind: Option<CompletionItemKind>,

    /// Lookup is used to check if completion item indeed can complete current
    /// ident.
    ///
    /// That is, in `foo.bar<|>` lookup of `abracadabra` will be accepted (it
    /// contains `bar` sub sequence), and `quux` will rejected.
    lookup: Option<String>,

    /// Additional info to show in the UI pop up.
    detail: Option<String>,
    documentation: Option<Documentation>,

    /// Whether this item is marked as deprecated
    deprecated: bool,

    /// If completing a function call, ask the editor to show parameter popup
    /// after completion.
    trigger_call_info: bool,

    /// Score is useful to pre select or display in better order completion items
    score: Option<CompletionScore>,
}

// We use custom debug for CompletionItem to make `insta`'s diffs more readable.
impl fmt::Debug for CompletionItem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = f.debug_struct("CompletionItem");
        s.field("label", &self.label()).field("source_range", &self.source_range());
        if self.text_edit().len() == 1 {
            let atom = &self.text_edit().iter().next().unwrap();
            s.field("delete", &atom.delete);
            s.field("insert", &atom.insert);
        } else {
            s.field("text_edit", &self.text_edit);
        }
        if let Some(kind) = self.kind().as_ref() {
            s.field("kind", kind);
        }
        if self.lookup() != self.label() {
            s.field("lookup", &self.lookup());
        }
        if let Some(detail) = self.detail() {
            s.field("detail", &detail);
        }
        if let Some(documentation) = self.documentation() {
            s.field("documentation", &documentation);
        }
        if self.deprecated {
            s.field("deprecated", &true);
        }
        if let Some(score) = &self.score {
            s.field("score", score);
        }
        if self.trigger_call_info {
            s.field("trigger_call_info", &true);
        }
        s.finish()
    }
}

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum CompletionScore {
    /// If only type match
    TypeMatch,
    /// If type and name match
    TypeAndNameMatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionItemKind {
    Snippet,
    Keyword,
    Module,
    Function,
    BuiltinType,
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
    TypeParam,
    Macro,
    Attribute,
}

impl CompletionItemKind {
    #[cfg(test)]
    pub(crate) fn tag(&self) -> &'static str {
        match self {
            CompletionItemKind::Attribute => "at",
            CompletionItemKind::Binding => "bn",
            CompletionItemKind::BuiltinType => "bt",
            CompletionItemKind::Const => "ct",
            CompletionItemKind::Enum => "en",
            CompletionItemKind::EnumVariant => "ev",
            CompletionItemKind::Field => "fd",
            CompletionItemKind::Function => "fn",
            CompletionItemKind::Keyword => "kw",
            CompletionItemKind::Macro => "ma",
            CompletionItemKind::Method => "me",
            CompletionItemKind::Module => "md",
            CompletionItemKind::Snippet => "sn",
            CompletionItemKind::Static => "sc",
            CompletionItemKind::Struct => "st",
            CompletionItemKind::Trait => "tt",
            CompletionItemKind::TypeAlias => "ta",
            CompletionItemKind::TypeParam => "tp",
        }
    }
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
    BuiltinType,
    Attribute,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum InsertTextFormat {
    PlainText,
    Snippet,
}

impl CompletionItem {
    pub(crate) fn new(
        completion_kind: CompletionKind,
        source_range: TextRange,
        label: impl Into<String>,
    ) -> Builder {
        let label = label.into();
        Builder {
            source_range,
            completion_kind,
            label,
            insert_text: None,
            insert_text_format: InsertTextFormat::PlainText,
            detail: None,
            documentation: None,
            lookup: None,
            kind: None,
            text_edit: None,
            deprecated: None,
            trigger_call_info: None,
            score: None,
        }
    }
    /// What user sees in pop-up in the UI.
    pub fn label(&self) -> &str {
        &self.label
    }
    pub fn source_range(&self) -> TextRange {
        self.source_range
    }

    pub fn insert_text_format(&self) -> InsertTextFormat {
        self.insert_text_format
    }

    pub fn text_edit(&self) -> &TextEdit {
        &self.text_edit
    }

    /// Short one-line additional information, like a type
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
    /// A doc-comment
    pub fn documentation(&self) -> Option<Documentation> {
        self.documentation.clone()
    }
    /// What string is used for filtering.
    pub fn lookup(&self) -> &str {
        self.lookup.as_deref().unwrap_or(&self.label)
    }

    pub fn kind(&self) -> Option<CompletionItemKind> {
        self.kind
    }

    pub fn deprecated(&self) -> bool {
        self.deprecated
    }

    pub fn score(&self) -> Option<CompletionScore> {
        self.score
    }

    pub fn trigger_call_info(&self) -> bool {
        self.trigger_call_info
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
    documentation: Option<Documentation>,
    lookup: Option<String>,
    kind: Option<CompletionItemKind>,
    text_edit: Option<TextEdit>,
    deprecated: Option<bool>,
    trigger_call_info: Option<bool>,
    score: Option<CompletionScore>,
}

impl Builder {
    pub(crate) fn add_to(self, acc: &mut Completions) {
        acc.add(self.build())
    }

    pub(crate) fn build(self) -> CompletionItem {
        let label = self.label;
        let text_edit = match self.text_edit {
            Some(it) => it,
            None => TextEdit::replace(
                self.source_range,
                self.insert_text.unwrap_or_else(|| label.clone()),
            ),
        };

        CompletionItem {
            source_range: self.source_range,
            label,
            insert_text_format: self.insert_text_format,
            text_edit,
            detail: self.detail,
            documentation: self.documentation,
            lookup: self.lookup,
            kind: self.kind,
            completion_kind: self.completion_kind,
            deprecated: self.deprecated.unwrap_or(false),
            trigger_call_info: self.trigger_call_info.unwrap_or(false),
            score: self.score,
        }
    }
    pub(crate) fn lookup_by(mut self, lookup: impl Into<String>) -> Builder {
        self.lookup = Some(lookup.into());
        self
    }
    pub(crate) fn label(mut self, label: impl Into<String>) -> Builder {
        self.label = label.into();
        self
    }
    pub(crate) fn insert_text(mut self, insert_text: impl Into<String>) -> Builder {
        self.insert_text = Some(insert_text.into());
        self
    }
    pub(crate) fn insert_snippet(
        mut self,
        _cap: SnippetCap,
        snippet: impl Into<String>,
    ) -> Builder {
        self.insert_text_format = InsertTextFormat::Snippet;
        self.insert_text(snippet)
    }
    pub(crate) fn kind(mut self, kind: CompletionItemKind) -> Builder {
        self.kind = Some(kind);
        self
    }
    pub(crate) fn text_edit(mut self, edit: TextEdit) -> Builder {
        self.text_edit = Some(edit);
        self
    }
    pub(crate) fn snippet_edit(mut self, _cap: SnippetCap, edit: TextEdit) -> Builder {
        self.insert_text_format = InsertTextFormat::Snippet;
        self.text_edit(edit)
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
    pub(crate) fn documentation(self, docs: Documentation) -> Builder {
        self.set_documentation(Some(docs))
    }
    pub(crate) fn set_documentation(mut self, docs: Option<Documentation>) -> Builder {
        self.documentation = docs.map(Into::into);
        self
    }
    pub(crate) fn set_deprecated(mut self, deprecated: bool) -> Builder {
        self.deprecated = Some(deprecated);
        self
    }
    pub(crate) fn set_score(mut self, score: CompletionScore) -> Builder {
        self.score = Some(score);
        self
    }
    pub(crate) fn trigger_call_info(mut self) -> Builder {
        self.trigger_call_info = Some(true);
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
