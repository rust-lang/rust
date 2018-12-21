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
