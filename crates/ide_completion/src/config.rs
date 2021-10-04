//! Settings for tweaking completion.
//!
//! The fun thing here is `SnippetCap` -- this type can only be created in this
//! module, and we use to statically check that we only produce snippet
//! completions if we are allowed to.

use ide_db::helpers::{insert_use::InsertUseConfig, SnippetCap};
use itertools::Itertools;
use syntax::ast;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionConfig {
    pub enable_postfix_completions: bool,
    pub enable_imports_on_the_fly: bool,
    pub enable_self_on_the_fly: bool,
    pub add_call_parenthesis: bool,
    pub add_call_argument_snippets: bool,
    pub snippet_cap: Option<SnippetCap>,
    pub insert_use: InsertUseConfig,
    pub postfix_snippets: Vec<PostfixSnippet>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PostfixSnippet {
    pub label: String,
    snippet: String,
    pub description: Option<String>,
    pub requires: Box<[String]>,
}

impl PostfixSnippet {
    pub fn new(
        label: String,
        snippet: &[String],
        description: &[String],
        requires: &[String],
    ) -> Option<Self> {
        // validate that these are indeed simple paths
        if requires.iter().any(|path| match ast::Path::parse(path) {
            Ok(path) => path.segments().any(|seg| {
                !matches!(seg.kind(), Some(ast::PathSegmentKind::Name(_)))
                    || seg.generic_arg_list().is_some()
            }),
            Err(_) => true,
        }) {
            return None;
        }
        let snippet = snippet.iter().join("\n");
        let description = description.iter().join("\n");
        let description = if description.is_empty() { None } else { Some(description) };
        Some(PostfixSnippet {
            label,
            snippet,
            description,
            requires: requires.iter().cloned().collect(), // Box::into doesn't work as that has a Copy bound 😒
        })
    }

    pub fn snippet(&self, receiver: &str) -> String {
        self.snippet.replace("$receiver", receiver)
    }
}
