//! User (postfix)-snippet definitions.
//!
//! Actual logic is implemented in [`crate::completions::postfix`] and [`crate::completions::snippet`] respectively.

// Feature: User Snippet Completions
//
// rust-analyzer allows the user to define custom (postfix)-snippets that may depend on items to be accessible for the current scope to be applicable.
//
// A custom snippet can be defined by adding it to the `rust-analyzer.completion.snippets.custom` object respectively.
//
// [source,json]
// ----
// {
//   "rust-analyzer.completion.snippets.custom": {
//     "thread spawn": {
//       "prefix": ["spawn", "tspawn"],
//       "body": [
//         "thread::spawn(move || {",
//         "\t$0",
//         "});",
//       ],
//       "description": "Insert a thread::spawn call",
//       "requires": "std::thread",
//       "scope": "expr",
//     }
//   }
// }
// ----
//
// In the example above:
//
// * `"thread spawn"` is the name of the snippet.
//
// * `prefix` defines one or more trigger words that will trigger the snippets completion.
// Using `postfix` will instead create a postfix snippet.
//
// * `body` is one or more lines of content joined via newlines for the final output.
//
// * `description` is an optional description of the snippet, if unset the snippet name will be used.
//
// * `requires` is an optional list of item paths that have to be resolvable in the current crate where the completion is rendered.
// On failure of resolution the snippet won't be applicable, otherwise the snippet will insert an import for the items on insertion if
// the items aren't yet in scope.
//
// * `scope` is an optional filter for when the snippet should be applicable. Possible values are:
// ** for Snippet-Scopes: `expr`, `item` (default: `item`)
// ** for Postfix-Snippet-Scopes: `expr`, `type` (default: `expr`)
//
// The `body` field also has access to placeholders as visible in the example as `$0`.
// These placeholders take the form of `$number` or `${number:placeholder_text}` which can be traversed as tabstop in ascending order starting from 1,
// with `$0` being a special case that always comes last.
//
// There is also a special placeholder, `${receiver}`, which will be replaced by the receiver expression for postfix snippets, or a `$0` tabstop in case of normal snippets.
// This replacement for normal snippets allows you to reuse a snippet for both post- and prefix in a single definition.
//
// For the VSCode editor, rust-analyzer also ships with a small set of defaults which can be removed
// by overwriting the settings object mentioned above, the defaults are:
// [source,json]
// ----
// {
//     "Arc::new": {
//         "postfix": "arc",
//         "body": "Arc::new(${receiver})",
//         "requires": "std::sync::Arc",
//         "description": "Put the expression into an `Arc`",
//         "scope": "expr"
//     },
//     "Rc::new": {
//         "postfix": "rc",
//         "body": "Rc::new(${receiver})",
//         "requires": "std::rc::Rc",
//         "description": "Put the expression into an `Rc`",
//         "scope": "expr"
//     },
//     "Box::pin": {
//         "postfix": "pinbox",
//         "body": "Box::pin(${receiver})",
//         "requires": "std::boxed::Box",
//         "description": "Put the expression into a pinned `Box`",
//         "scope": "expr"
//     },
//     "Ok": {
//         "postfix": "ok",
//         "body": "Ok(${receiver})",
//         "description": "Wrap the expression in a `Result::Ok`",
//         "scope": "expr"
//     },
//     "Err": {
//         "postfix": "err",
//         "body": "Err(${receiver})",
//         "description": "Wrap the expression in a `Result::Err`",
//         "scope": "expr"
//     },
//     "Some": {
//         "postfix": "some",
//         "body": "Some(${receiver})",
//         "description": "Wrap the expression in an `Option::Some`",
//         "scope": "expr"
//     }
// }
// ----

use ide_db::imports::import_assets::LocatedImport;
use itertools::Itertools;
use syntax::{ast, AstNode, GreenNode, SyntaxNode};

use crate::context::CompletionContext;

/// A snippet scope describing where a snippet may apply to.
/// These may differ slightly in meaning depending on the snippet trigger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnippetScope {
    Item,
    Expr,
    Type,
}

/// A user supplied snippet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snippet {
    pub postfix_triggers: Box<[Box<str>]>,
    pub prefix_triggers: Box<[Box<str>]>,
    pub scope: SnippetScope,
    pub description: Option<Box<str>>,
    snippet: String,
    // These are `ast::Path`'s but due to SyntaxNodes not being Send we store these
    // and reconstruct them on demand instead. This is cheaper than reparsing them
    // from strings
    requires: Box<[GreenNode]>,
}

impl Snippet {
    pub fn new(
        prefix_triggers: &[String],
        postfix_triggers: &[String],
        snippet: &[String],
        description: &str,
        requires: &[String],
        scope: SnippetScope,
    ) -> Option<Self> {
        if prefix_triggers.is_empty() && postfix_triggers.is_empty() {
            return None;
        }
        let (requires, snippet, description) = validate_snippet(snippet, description, requires)?;
        Some(Snippet {
            // Box::into doesn't work as that has a Copy bound 😒
            postfix_triggers: postfix_triggers.iter().map(String::as_str).map(Into::into).collect(),
            prefix_triggers: prefix_triggers.iter().map(String::as_str).map(Into::into).collect(),
            scope,
            snippet,
            description,
            requires,
        })
    }

    /// Returns [`None`] if the required items do not resolve.
    pub(crate) fn imports(&self, ctx: &CompletionContext<'_>) -> Option<Vec<LocatedImport>> {
        import_edits(ctx, &self.requires)
    }

    pub fn snippet(&self) -> String {
        self.snippet.replace("${receiver}", "$0")
    }

    pub fn postfix_snippet(&self, receiver: &str) -> String {
        self.snippet.replace("${receiver}", receiver)
    }
}

fn import_edits(ctx: &CompletionContext<'_>, requires: &[GreenNode]) -> Option<Vec<LocatedImport>> {
    let resolve = |import: &GreenNode| {
        let path = ast::Path::cast(SyntaxNode::new_root(import.clone()))?;
        let item = match ctx.scope.speculative_resolve(&path)? {
            hir::PathResolution::Def(def) => def.into(),
            _ => return None,
        };
        let path = ctx.module.find_use_path_prefixed(
            ctx.db,
            item,
            ctx.config.insert_use.prefix_kind,
            ctx.config.prefer_no_std,
            ctx.config.prefer_prelude,
        )?;
        Some((path.len() > 1).then(|| LocatedImport::new(path.clone(), item, item, None)))
    };
    let mut res = Vec::with_capacity(requires.len());
    for import in requires {
        match resolve(import) {
            Some(first) => res.extend(first),
            None => return None,
        }
    }
    Some(res)
}

fn validate_snippet(
    snippet: &[String],
    description: &str,
    requires: &[String],
) -> Option<(Box<[GreenNode]>, String, Option<Box<str>>)> {
    let mut imports = Vec::with_capacity(requires.len());
    for path in requires.iter() {
        let use_path = ast::SourceFile::parse(&format!("use {path};"))
            .syntax_node()
            .descendants()
            .find_map(ast::Path::cast)?;
        if use_path.syntax().text() != path.as_str() {
            return None;
        }
        let green = use_path.syntax().green().into_owned();
        imports.push(green);
    }
    let snippet = snippet.iter().join("\n");
    let description = (!description.is_empty())
        .then(|| description.split_once('\n').map_or(description, |(it, _)| it))
        .map(ToOwned::to_owned)
        .map(Into::into);
    Some((imports.into_boxed_slice(), snippet, description))
}
