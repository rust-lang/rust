/// `CompletionItem` describes a single completion variant in the editor pop-up.
/// It is basically a POD with various properties. To construct a
/// `CompletionItem`, use `new` method and the `Builder` struct.
#[derive(Debug)]
pub struct CompletionItem {
    label: String,
    lookup: Option<String>,
    snippet: Option<String>,
}

pub enum InsertText {
    PlainText { text: String },
    Snippet { text: String },
}

impl CompletionItem {
    pub(crate) fn new(label: impl Into<String>) -> Builder {
        let label = label.into();
        Builder {
            label,
            lookup: None,
            snippet: None,
        }
    }
    /// What user sees in pop-up in the UI.
    pub fn label(&self) -> &str {
        &self.label
    }
    /// What string is used for filtering.
    pub fn lookup(&self) -> &str {
        self.lookup
            .as_ref()
            .map(|it| it.as_str())
            .unwrap_or(self.label())
    }
    /// What is inserted.
    pub fn insert_text(&self) -> InsertText {
        match &self.snippet {
            None => InsertText::PlainText {
                text: self.label.clone(),
            },
            Some(it) => InsertText::Snippet { text: it.clone() },
        }
    }
}

/// A helper to make `CompletionItem`s.
#[must_use]
pub(crate) struct Builder {
    label: String,
    lookup: Option<String>,
    snippet: Option<String>,
}

impl Builder {
    pub fn add_to(self, acc: &mut Vec<CompletionItem>) {
        acc.push(self.build())
    }

    pub fn build(self) -> CompletionItem {
        CompletionItem {
            label: self.label,
            lookup: self.lookup,
            snippet: self.snippet,
        }
    }
    pub fn lookup_by(mut self, lookup: impl Into<String>) -> Builder {
        self.lookup = Some(lookup.into());
        self
    }
    pub fn snippet(mut self, snippet: impl Into<String>) -> Builder {
        self.snippet = Some(snippet.into());
        self
    }
}

impl Into<CompletionItem> for Builder {
    fn into(self) -> CompletionItem {
        self.build()
    }
}

/// Represents an in-progress set of completions being built.
#[derive(Debug)]
pub(crate) struct Completions {
    buf: Vec<CompletionItem>,
}

impl Completions {
    pub(crate) fn add(&mut self, item: impl Into<CompletionItem>) {
        self.buf.push(item.into())
    }
}

impl Into<Vec<CompletionItem>> for Completions {
    fn into(self) -> Vec<CompletionItem> {
        self.buf
    }
}
