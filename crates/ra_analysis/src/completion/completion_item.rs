#[derive(Debug)]
pub struct CompletionItem {
    /// What user sees in pop-up in the UI.
    pub label: String,
    /// What string is used for filtering, defaults to label.
    pub lookup: Option<String>,
    /// What is inserted, defaults to label.
    pub snippet: Option<String>,
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
}

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
}
