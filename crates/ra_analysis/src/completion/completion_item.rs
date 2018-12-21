/// `CompletionItem` describes a single completion variant in the editor pop-up.
/// It is basically a POD with various properties. To construct a
/// `CompletionItem`, use `new` method and the `Builder` struct.
#[derive(Debug)]
pub struct CompletionItem {
    label: String,
    lookup: Option<String>,
    snippet: Option<String>,
    kind: Option<CompletionItemKind>,
    /// Used only internally in tests, to check only specific kind of
    /// completion.
    completion_kind: CompletionKind,
}

pub enum InsertText {
    PlainText { text: String },
    Snippet { text: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionItemKind {
    Snippet,
    Keyword,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CompletionKind {
    /// Parser-based keyword completion.
    Keyword,
    /// Your usual "complete all valid identifiers".
    Reference,
    /// "Secret sauce" completions.
    Magic,
    Snippet,
}

impl CompletionItem {
    pub(crate) fn new(completion_kind: CompletionKind, label: impl Into<String>) -> Builder {
        let label = label.into();
        Builder {
            label,
            lookup: None,
            snippet: None,
            completion_kind,
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

    pub fn kind(&self) -> Option<CompletionItemKind> {
        self.kind
    }
}

/// A helper to make `CompletionItem`s.
#[must_use]
pub(crate) struct Builder {
    label: String,
    lookup: Option<String>,
    snippet: Option<String>,
    completion_kind: CompletionKind,
}

impl Builder {
    pub(crate) fn add_to(self, acc: &mut Completions) {
        acc.add(self.build())
    }

    pub(crate) fn build(self) -> CompletionItem {
        CompletionItem {
            label: self.label,
            lookup: self.lookup,
            snippet: self.snippet,
            kind: None,
            completion_kind: self.completion_kind,
        }
    }
    pub(crate) fn lookup_by(mut self, lookup: impl Into<String>) -> Builder {
        self.lookup = Some(lookup.into());
        self
    }
    pub(crate) fn snippet(mut self, snippet: impl Into<String>) -> Builder {
        self.snippet = Some(snippet.into());
        self
    }
    pub(crate) fn kind(mut self, kind: CompletionKind) -> Builder {
        self.completion_kind = kind;
        self
    }
}

impl Into<CompletionItem> for Builder {
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

    #[cfg(test)]
    pub(crate) fn assert_match(&self, expected: &str, kind: CompletionKind) {
        let expected = normalize(expected);
        let actual = self.debug_render(kind);
        test_utils::assert_eq_text!(expected.as_str(), actual.as_str(),);

        /// Normalize the textual representation of `Completions`:
        /// replace `;` with newlines, normalize whitespace
        fn normalize(expected: &str) -> String {
            use ra_syntax::{tokenize, TextUnit, TextRange, SyntaxKind::SEMI};
            let mut res = String::new();
            for line in expected.trim().lines() {
                let line = line.trim();
                let mut start_offset: TextUnit = 0.into();
                // Yep, we use rust tokenize in completion tests :-)
                for token in tokenize(line) {
                    let range = TextRange::offset_len(start_offset, token.len);
                    start_offset += token.len;
                    if token.kind == SEMI {
                        res.push('\n');
                    } else {
                        res.push_str(&line[range]);
                    }
                }

                res.push('\n');
            }
            res
        }
    }

    #[cfg(test)]
    fn debug_render(&self, kind: CompletionKind) -> String {
        let mut res = String::new();
        for c in self.buf.iter() {
            if c.completion_kind == kind {
                if let Some(lookup) = &c.lookup {
                    res.push_str(lookup);
                    res.push_str(&format!(" {:?}", c.label));
                } else {
                    res.push_str(&c.label);
                }
                if let Some(snippet) = &c.snippet {
                    res.push_str(&format!(" {:?}", snippet));
                }
                res.push('\n');
            }
        }
        res
    }
}

impl Into<Vec<CompletionItem>> for Completions {
    fn into(self) -> Vec<CompletionItem> {
        self.buf
    }
}
