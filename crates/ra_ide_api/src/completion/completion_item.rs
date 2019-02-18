use std::fmt;

use hir::{Docs, Documentation, PerNs, Resolution};
use ra_syntax::TextRange;
use ra_text_edit::{ TextEditBuilder, TextEdit};
use test_utils::tested_by;

use crate::completion::{
    completion_context::CompletionContext,
    function_label,
    const_label,
    type_label
};

/// `CompletionItem` describes a single completion variant in the editor pop-up.
/// It is basically a POD with various properties. To construct a
/// `CompletionItem`, use `new` method and the `Builder` struct.
pub struct CompletionItem {
    /// Used only internally in tests, to check only specific kind of
    /// completion (postfix, keyword, reference, etc).
    #[allow(unused)]
    completion_kind: CompletionKind,
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
}

// We use custom debug for CompletionItem to make `insta`'s diffs more readable.
impl fmt::Debug for CompletionItem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = f.debug_struct("CompletionItem");
        s.field("label", &self.label()).field("source_range", &self.source_range());
        if self.text_edit().as_atoms().len() == 1 {
            let atom = &self.text_edit().as_atoms()[0];
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
        s.finish()
    }
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
    TypeParam,
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
        self.detail.as_ref().map(|it| it.as_str())
    }
    /// A doc-comment
    pub fn documentation(&self) -> Option<Documentation> {
        self.documentation.clone()
    }
    /// What string is used for filtering.
    pub fn lookup(&self) -> &str {
        self.lookup.as_ref().map(|it| it.as_str()).unwrap_or_else(|| self.label())
    }

    pub fn kind(&self) -> Option<CompletionItemKind> {
        self.kind
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
}

impl Builder {
    pub(crate) fn add_to(self, acc: &mut Completions) {
        acc.add(self.build())
    }

    pub(crate) fn build(self) -> CompletionItem {
        let label = self.label;
        let text_edit = match self.text_edit {
            Some(it) => it,
            None => {
                let mut builder = TextEditBuilder::default();
                builder
                    .replace(self.source_range, self.insert_text.unwrap_or_else(|| label.clone()));
                builder.finish()
            }
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
    pub(crate) fn insert_snippet(mut self, snippet: impl Into<String>) -> Builder {
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
    pub(crate) fn snippet_edit(mut self, edit: TextEdit) -> Builder {
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
    pub(super) fn from_resolution(
        mut self,
        ctx: &CompletionContext,
        resolution: &PerNs<Resolution>,
    ) -> Builder {
        use hir::ModuleDef::*;

        let def = resolution.as_ref().take_types().or_else(|| resolution.as_ref().take_values());
        let def = match def {
            None => return self,
            Some(it) => it,
        };
        let (kind, docs) = match def {
            Resolution::Def(Module(it)) => (CompletionItemKind::Module, it.docs(ctx.db)),
            Resolution::Def(Function(func)) => return self.from_function(ctx, *func),
            Resolution::Def(Struct(it)) => (CompletionItemKind::Struct, it.docs(ctx.db)),
            Resolution::Def(Enum(it)) => (CompletionItemKind::Enum, it.docs(ctx.db)),
            Resolution::Def(EnumVariant(it)) => (CompletionItemKind::EnumVariant, it.docs(ctx.db)),
            Resolution::Def(Const(it)) => (CompletionItemKind::Const, it.docs(ctx.db)),
            Resolution::Def(Static(it)) => (CompletionItemKind::Static, it.docs(ctx.db)),
            Resolution::Def(Trait(it)) => (CompletionItemKind::Trait, it.docs(ctx.db)),
            Resolution::Def(Type(it)) => (CompletionItemKind::TypeAlias, it.docs(ctx.db)),
            Resolution::GenericParam(..) => (CompletionItemKind::TypeParam, None),
            Resolution::LocalBinding(..) => (CompletionItemKind::Binding, None),
            Resolution::SelfType(..) => (
                CompletionItemKind::TypeParam, // (does this need its own kind?)
                None,
            ),
        };
        self.kind = Some(kind);
        self.documentation = docs;

        self
    }

    pub(super) fn from_function(
        mut self,
        ctx: &CompletionContext,
        function: hir::Function,
    ) -> Builder {
        // If not an import, add parenthesis automatically.
        if ctx.use_item_syntax.is_none() && !ctx.is_call {
            tested_by!(inserts_parens_for_function_calls);
            let sig = function.signature(ctx.db);
            if sig.params().is_empty() || sig.has_self_param() && sig.params().len() == 1 {
                self.insert_text = Some(format!("{}()$0", self.label));
            } else {
                self.insert_text = Some(format!("{}($0)", self.label));
            }
            self.insert_text_format = InsertTextFormat::Snippet;
        }

        if let Some(docs) = function.docs(ctx.db) {
            self.documentation = Some(docs);
        }

        if let Some(label) = function_item_label(ctx, function) {
            self.detail = Some(label);
        }

        self.kind = Some(CompletionItemKind::Function);
        self
    }

    pub(super) fn from_const(mut self, ctx: &CompletionContext, ct: hir::Const) -> Builder {
        if let Some(docs) = ct.docs(ctx.db) {
            self.documentation = Some(docs);
        }

        self.detail = Some(const_item_label(ctx, ct));
        self.kind = Some(CompletionItemKind::Const);

        self
    }

    pub(super) fn from_type(mut self, ctx: &CompletionContext, ty: hir::Type) -> Builder {
        if let Some(docs) = ty.docs(ctx.db) {
            self.documentation = Some(docs);
        }

        self.detail = Some(type_item_label(ctx, ty));
        self.kind = Some(CompletionItemKind::TypeAlias);

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

fn function_item_label(ctx: &CompletionContext, function: hir::Function) -> Option<String> {
    let node = function.source(ctx.db).1;
    function_label(&node)
}

fn const_item_label(ctx: &CompletionContext, ct: hir::Const) -> String {
    let node = ct.source(ctx.db).1;
    const_label(&node)
}

fn type_item_label(ctx: &CompletionContext, ty: hir::Type) -> String {
    let node = ty.source(ctx.db).1;
    type_label(&node)
}

#[cfg(test)]
pub(crate) fn do_completion(code: &str, kind: CompletionKind) -> Vec<CompletionItem> {
    use crate::mock_analysis::{single_file_with_position, analysis_and_position};
    use crate::completion::completions;
    let (analysis, position) = if code.contains("//-") {
        analysis_and_position(code)
    } else {
        single_file_with_position(code)
    };
    let completions = completions(&analysis.db, position).unwrap();
    let completion_items: Vec<CompletionItem> = completions.into();
    let mut kind_completions: Vec<CompletionItem> =
        completion_items.into_iter().filter(|c| c.completion_kind == kind).collect();
    kind_completions.sort_by_key(|c| c.label.clone());
    kind_completions
}

#[cfg(test)]
pub(crate) fn check_completion(test_name: &str, code: &str, kind: CompletionKind) {
    use insta::assert_debug_snapshot_matches;
    let kind_completions = do_completion(code, kind);
    assert_debug_snapshot_matches!(test_name, kind_completions);
}

#[cfg(test)]
mod tests {
    use test_utils::covers;

    use super::*;

    fn check_reference_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Reference);
    }

    #[test]
    fn inserts_parens_for_function_calls() {
        covers!(inserts_parens_for_function_calls);
        check_reference_completion(
            "inserts_parens_for_function_calls1",
            r"
            fn no_args() {}
            fn main() { no_<|> }
            ",
        );
        check_reference_completion(
            "inserts_parens_for_function_calls2",
            r"
            fn with_args(x: i32, y: String) {}
            fn main() { with_<|> }
            ",
        );
        check_reference_completion(
            "inserts_parens_for_function_calls3",
            r"
            struct S {}
            impl S {
                fn foo(&self) {}
            }
            fn bar(s: &S) {
                s.f<|>
            }
            ",
        )
    }

    #[test]
    fn dont_render_function_parens_in_use_item() {
        check_reference_completion(
            "dont_render_function_parens_in_use_item",
            "
            //- /lib.rs
            mod m { pub fn foo() {} }
            use crate::m::f<|>;
            ",
        )
    }

    #[test]
    fn dont_render_function_parens_if_already_call() {
        check_reference_completion(
            "dont_render_function_parens_if_already_call",
            "
            //- /lib.rs
            fn frobnicate() {}
            fn main() {
                frob<|>();
            }
            ",
        );
        check_reference_completion(
            "dont_render_function_parens_if_already_call_assoc_fn",
            "
            //- /lib.rs
            struct Foo {}
            impl Foo { fn new() -> Foo {} }
            fn main() {
                Foo::ne<|>();
            }
            ",
        )
    }

}
