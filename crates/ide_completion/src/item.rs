//! See `CompletionItem` structure.

use std::fmt;

use hir::{Documentation, Mutability};
use ide_db::{
    helpers::{
        import_assets::LocatedImport,
        insert_use::{self, ImportScope, InsertUseConfig},
        mod_path_to_ast, SnippetCap,
    },
    SymbolKind,
};
use stdx::{format_to, impl_from, never};
use syntax::{algo, TextRange};
use text_edit::TextEdit;

/// `CompletionItem` describes a single completion variant in the editor pop-up.
/// It is basically a POD with various properties. To construct a
/// `CompletionItem`, use `new` method and the `Builder` struct.
#[derive(Clone)]
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
    /// That is, in `foo.bar$0` lookup of `abracadabra` will be accepted (it
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

    /// We use this to sort completion. Relevance records facts like "do the
    /// types align precisely?". We can't sort by relevances directly, they are
    /// only partially ordered.
    ///
    /// Note that Relevance ignores fuzzy match score. We compute Relevance for
    /// all possible items, and then separately build an ordered completion list
    /// based on relevance and fuzzy matching with the already typed identifier.
    relevance: CompletionRelevance,

    /// Indicates that a reference or mutable reference to this variable is a
    /// possible match.
    ref_match: Option<Mutability>,

    /// The import data to add to completion's edits.
    import_to_add: Option<ImportEdit>,
}

// We use custom debug for CompletionItem to make snapshot tests more readable.
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

        if self.relevance != CompletionRelevance::default() {
            s.field("relevance", &self.relevance);
        }

        if let Some(mutability) = &self.ref_match {
            s.field("ref_match", &format!("&{}", mutability.as_keyword_for_ref()));
        }
        if self.trigger_call_info {
            s.field("trigger_call_info", &true);
        }
        s.finish()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub struct CompletionRelevance {
    /// This is set in cases like these:
    ///
    /// ```
    /// fn f(spam: String) {}
    /// fn main {
    ///     let spam = 92;
    ///     f($0) // name of local matches the name of param
    /// }
    /// ```
    pub exact_name_match: bool,
    /// See CompletionRelevanceTypeMatch doc comments for cases where this is set.
    pub type_match: Option<CompletionRelevanceTypeMatch>,
    /// This is set in cases like these:
    ///
    /// ```
    /// fn foo(a: u32) {
    ///     let b = 0;
    ///     $0 // `a` and `b` are local
    /// }
    /// ```
    pub is_local: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CompletionRelevanceTypeMatch {
    /// This is set in cases like these:
    ///
    /// ```
    /// enum Option<T> { Some(T), None }
    /// fn f(a: Option<u32>) {}
    /// fn main {
    ///     f(Option::N$0) // type `Option<T>` could unify with `Option<u32>`
    /// }
    /// ```
    CouldUnify,
    /// This is set in cases like these:
    ///
    /// ```
    /// fn f(spam: String) {}
    /// fn main {
    ///     let foo = String::new();
    ///     f($0) // type of local matches the type of param
    /// }
    /// ```
    Exact,
}

impl CompletionRelevance {
    /// Provides a relevance score. Higher values are more relevant.
    ///
    /// The absolute value of the relevance score is not meaningful, for
    /// example a value of 0 doesn't mean "not relevant", rather
    /// it means "least relevant". The score value should only be used
    /// for relative ordering.
    ///
    /// See is_relevant if you need to make some judgement about score
    /// in an absolute sense.
    pub fn score(&self) -> u32 {
        let mut score = 0;

        if self.exact_name_match {
            score += 1;
        }
        score += match self.type_match {
            Some(CompletionRelevanceTypeMatch::Exact) => 4,
            Some(CompletionRelevanceTypeMatch::CouldUnify) => 3,
            None => 0,
        };
        if self.is_local {
            score += 1;
        }

        score
    }

    /// Returns true when the score for this threshold is above
    /// some threshold such that we think it is especially likely
    /// to be relevant.
    pub fn is_relevant(&self) -> bool {
        self.score() > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionItemKind {
    SymbolKind(SymbolKind),
    Attribute,
    Binding,
    BuiltinType,
    Keyword,
    Method,
    Snippet,
    UnresolvedReference,
}

impl_from!(SymbolKind for CompletionItemKind);

impl CompletionItemKind {
    #[cfg(test)]
    pub(crate) fn tag(&self) -> &'static str {
        match self {
            CompletionItemKind::SymbolKind(kind) => match kind {
                SymbolKind::Const => "ct",
                SymbolKind::ConstParam => "cp",
                SymbolKind::Enum => "en",
                SymbolKind::Field => "fd",
                SymbolKind::Function => "fn",
                SymbolKind::Impl => "im",
                SymbolKind::Label => "lb",
                SymbolKind::LifetimeParam => "lt",
                SymbolKind::Local => "lc",
                SymbolKind::Macro => "ma",
                SymbolKind::Module => "md",
                SymbolKind::SelfParam => "sp",
                SymbolKind::Static => "sc",
                SymbolKind::Struct => "st",
                SymbolKind::Trait => "tt",
                SymbolKind::TypeAlias => "ta",
                SymbolKind::TypeParam => "tp",
                SymbolKind::Union => "un",
                SymbolKind::ValueParam => "vp",
                SymbolKind::Variant => "ev",
            },
            CompletionItemKind::Attribute => "at",
            CompletionItemKind::Binding => "bn",
            CompletionItemKind::BuiltinType => "bt",
            CompletionItemKind::Keyword => "kw",
            CompletionItemKind::Method => "me",
            CompletionItemKind::Snippet => "sn",
            CompletionItemKind::UnresolvedReference => "??",
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
            deprecated: false,
            trigger_call_info: None,
            relevance: CompletionRelevance::default(),
            ref_match: None,
            import_to_add: None,
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

    pub fn relevance(&self) -> CompletionRelevance {
        self.relevance
    }

    pub fn trigger_call_info(&self) -> bool {
        self.trigger_call_info
    }

    pub fn ref_match(&self) -> Option<(Mutability, CompletionRelevance)> {
        // Relevance of the ref match should be the same as the original
        // match, but with exact type match set because self.ref_match
        // is only set if there is an exact type match.
        let mut relevance = self.relevance;
        relevance.type_match = Some(CompletionRelevanceTypeMatch::Exact);

        self.ref_match.map(|mutability| (mutability, relevance))
    }

    pub fn import_to_add(&self) -> Option<&ImportEdit> {
        self.import_to_add.as_ref()
    }
}

/// An extra import to add after the completion is applied.
#[derive(Debug, Clone)]
pub struct ImportEdit {
    pub import: LocatedImport,
    pub scope: ImportScope,
}

impl ImportEdit {
    /// Attempts to insert the import to the given scope, producing a text edit.
    /// May return no edit in edge cases, such as scope already containing the import.
    pub fn to_text_edit(&self, cfg: InsertUseConfig) -> Option<TextEdit> {
        let _p = profile::span("ImportEdit::to_text_edit");

        let rewriter =
            insert_use::insert_use(&self.scope, mod_path_to_ast(&self.import.import_path), cfg);
        let old_ast = rewriter.rewrite_root()?;
        let mut import_insert = TextEdit::builder();
        algo::diff(&old_ast, &rewriter.rewrite(&old_ast)).into_text_edit(&mut import_insert);

        Some(import_insert.finish())
    }
}

/// A helper to make `CompletionItem`s.
#[must_use]
#[derive(Clone)]
pub(crate) struct Builder {
    source_range: TextRange,
    completion_kind: CompletionKind,
    import_to_add: Option<ImportEdit>,
    label: String,
    insert_text: Option<String>,
    insert_text_format: InsertTextFormat,
    detail: Option<String>,
    documentation: Option<Documentation>,
    lookup: Option<String>,
    kind: Option<CompletionItemKind>,
    text_edit: Option<TextEdit>,
    deprecated: bool,
    trigger_call_info: Option<bool>,
    relevance: CompletionRelevance,
    ref_match: Option<Mutability>,
}

impl Builder {
    pub(crate) fn build(self) -> CompletionItem {
        let _p = profile::span("item::Builder::build");

        let mut label = self.label;
        let mut lookup = self.lookup;
        let mut insert_text = self.insert_text;

        if let Some(original_path) = self
            .import_to_add
            .as_ref()
            .and_then(|import_edit| import_edit.import.original_path.as_ref())
        {
            lookup = lookup.or_else(|| Some(label.clone()));
            insert_text = insert_text.or_else(|| Some(label.clone()));

            let original_path_label = original_path.to_string();
            if original_path_label.ends_with(&label) {
                label = original_path_label;
            } else {
                format_to!(label, " ({})", original_path)
            }
        }

        let text_edit = match self.text_edit {
            Some(it) => it,
            None => {
                TextEdit::replace(self.source_range, insert_text.unwrap_or_else(|| label.clone()))
            }
        };

        CompletionItem {
            source_range: self.source_range,
            label,
            insert_text_format: self.insert_text_format,
            text_edit,
            detail: self.detail,
            documentation: self.documentation,
            lookup,
            kind: self.kind,
            completion_kind: self.completion_kind,
            deprecated: self.deprecated,
            trigger_call_info: self.trigger_call_info.unwrap_or(false),
            relevance: self.relevance,
            ref_match: self.ref_match,
            import_to_add: self.import_to_add,
        }
    }
    pub(crate) fn lookup_by(&mut self, lookup: impl Into<String>) -> &mut Builder {
        self.lookup = Some(lookup.into());
        self
    }
    pub(crate) fn label(&mut self, label: impl Into<String>) -> &mut Builder {
        self.label = label.into();
        self
    }
    pub(crate) fn insert_text(&mut self, insert_text: impl Into<String>) -> &mut Builder {
        self.insert_text = Some(insert_text.into());
        self
    }
    pub(crate) fn insert_snippet(
        &mut self,
        _cap: SnippetCap,
        snippet: impl Into<String>,
    ) -> &mut Builder {
        self.insert_text_format = InsertTextFormat::Snippet;
        self.insert_text(snippet)
    }
    pub(crate) fn kind(&mut self, kind: impl Into<CompletionItemKind>) -> &mut Builder {
        self.kind = Some(kind.into());
        self
    }
    pub(crate) fn text_edit(&mut self, edit: TextEdit) -> &mut Builder {
        self.text_edit = Some(edit);
        self
    }
    pub(crate) fn snippet_edit(&mut self, _cap: SnippetCap, edit: TextEdit) -> &mut Builder {
        self.insert_text_format = InsertTextFormat::Snippet;
        self.text_edit(edit)
    }
    pub(crate) fn detail(&mut self, detail: impl Into<String>) -> &mut Builder {
        self.set_detail(Some(detail))
    }
    pub(crate) fn set_detail(&mut self, detail: Option<impl Into<String>>) -> &mut Builder {
        self.detail = detail.map(Into::into);
        if let Some(detail) = &self.detail {
            if never!(detail.contains('\n'), "multiline detail:\n{}", detail) {
                self.detail = Some(detail.splitn(2, '\n').next().unwrap().to_string());
            }
        }
        self
    }
    #[allow(unused)]
    pub(crate) fn documentation(&mut self, docs: Documentation) -> &mut Builder {
        self.set_documentation(Some(docs))
    }
    pub(crate) fn set_documentation(&mut self, docs: Option<Documentation>) -> &mut Builder {
        self.documentation = docs.map(Into::into);
        self
    }
    pub(crate) fn set_deprecated(&mut self, deprecated: bool) -> &mut Builder {
        self.deprecated = deprecated;
        self
    }
    pub(crate) fn set_relevance(&mut self, relevance: CompletionRelevance) -> &mut Builder {
        self.relevance = relevance;
        self
    }
    pub(crate) fn trigger_call_info(&mut self) -> &mut Builder {
        self.trigger_call_info = Some(true);
        self
    }
    pub(crate) fn add_import(&mut self, import_to_add: Option<ImportEdit>) -> &mut Builder {
        self.import_to_add = import_to_add;
        self
    }
    pub(crate) fn ref_match(&mut self, mutability: Mutability) -> &mut Builder {
        self.ref_match = Some(mutability);
        self
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use test_utils::assert_eq_text;

    use super::{CompletionRelevance, CompletionRelevanceTypeMatch};

    /// Check that these are CompletionRelevance are sorted in ascending order
    /// by their relevance score.
    ///
    /// We want to avoid making assertions about the absolute score of any
    /// item, but we do want to assert whether each is >, <, or == to the
    /// others.
    ///
    /// If provided vec![vec![a], vec![b, c], vec![d]], then this will assert:
    ///     a.score < b.score == c.score < d.score
    fn check_relevance_score_ordered(expected_relevance_order: Vec<Vec<CompletionRelevance>>) {
        let expected = format!("{:#?}", &expected_relevance_order);

        let actual_relevance_order = expected_relevance_order
            .into_iter()
            .flatten()
            .map(|r| (r.score(), r))
            .sorted_by_key(|(score, _r)| *score)
            .fold(
                (u32::MIN, vec![vec![]]),
                |(mut currently_collecting_score, mut out), (score, r)| {
                    if currently_collecting_score == score {
                        out.last_mut().unwrap().push(r);
                    } else {
                        currently_collecting_score = score;
                        out.push(vec![r]);
                    }
                    (currently_collecting_score, out)
                },
            )
            .1;

        let actual = format!("{:#?}", &actual_relevance_order);

        assert_eq_text!(&expected, &actual);
    }

    #[test]
    fn relevance_score() {
        // This test asserts that the relevance score for these items is ascending, and
        // that any items in the same vec have the same score.
        let expected_relevance_order = vec![
            vec![CompletionRelevance::default()],
            vec![
                CompletionRelevance { exact_name_match: true, ..CompletionRelevance::default() },
                CompletionRelevance { is_local: true, ..CompletionRelevance::default() },
            ],
            vec![CompletionRelevance {
                exact_name_match: true,
                is_local: true,
                ..CompletionRelevance::default()
            }],
            vec![CompletionRelevance {
                type_match: Some(CompletionRelevanceTypeMatch::CouldUnify),
                ..CompletionRelevance::default()
            }],
            vec![CompletionRelevance {
                type_match: Some(CompletionRelevanceTypeMatch::Exact),
                ..CompletionRelevance::default()
            }],
            vec![CompletionRelevance {
                exact_name_match: true,
                type_match: Some(CompletionRelevanceTypeMatch::Exact),
                ..CompletionRelevance::default()
            }],
            vec![CompletionRelevance {
                exact_name_match: true,
                type_match: Some(CompletionRelevanceTypeMatch::Exact),
                is_local: true,
            }],
        ];

        check_relevance_score_ordered(expected_relevance_order);
    }
}
