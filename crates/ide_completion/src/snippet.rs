//! User (postfix)-snippet definitions.
//!
//! Actual logic is implemented in [`crate::completions::postfix`] and [`crate::completions::snippet`].

// Feature: User Snippet Completions
//
// rust-analyzer allows the user to define custom (postfix)-snippets that may depend on items to be accessible for the current scope to be applicable.
//
// A custom snippet can be defined by adding it to the `rust-analyzer.completion.snippets` object respectively.
//
// [source,json]
// ----
// {
//   "rust-analyzer.completion.snippets": {
//     "thread spawn": {
//       "prefix": ["spawn", "tspawn"],
//       "body": [
//         "thread::spawn(move || {",
//         "\t$0",
//         ")};",
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
// There is also a special placeholder, `${receiver}`, which will be replaced by the receiver expression for postfix snippets, or nothing in case of normal snippets.
// It does not act as a tabstop.
use ide_db::helpers::{import_assets::LocatedImport, insert_use::ImportScope};
use itertools::Itertools;
use syntax::ast;

use crate::{context::CompletionContext, ImportEdit};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnippetScope {
    Item,
    Expr,
    Type,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snippet {
    pub postfix_triggers: Box<[String]>,
    pub prefix_triggers: Box<[String]>,
    pub scope: SnippetScope,
    snippet: String,
    pub description: Option<String>,
    pub requires: Box<[String]>,
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
        let (snippet, description) = validate_snippet(snippet, description, requires)?;
        Some(Snippet {
            // Box::into doesn't work as that has a Copy bound 😒
            postfix_triggers: postfix_triggers.iter().cloned().collect(),
            prefix_triggers: prefix_triggers.iter().cloned().collect(),
            scope,
            snippet,
            description,
            requires: requires.iter().cloned().collect(),
        })
    }

    /// Returns None if the required items do not resolve.
    pub(crate) fn imports(
        &self,
        ctx: &CompletionContext,
        import_scope: &ImportScope,
    ) -> Option<Vec<ImportEdit>> {
        import_edits(ctx, import_scope, &self.requires)
    }

    pub fn snippet(&self) -> String {
        self.snippet.replace("${receiver}", "")
    }

    pub fn postfix_snippet(&self, receiver: &str) -> String {
        self.snippet.replace("${receiver}", receiver)
    }

    pub fn is_item(&self) -> bool {
        self.scope == SnippetScope::Item
    }

    pub fn is_expr(&self) -> bool {
        self.scope == SnippetScope::Expr
    }
}

fn import_edits(
    ctx: &CompletionContext,
    import_scope: &ImportScope,
    requires: &[String],
) -> Option<Vec<ImportEdit>> {
    let resolve = |import| {
        let path = ast::Path::parse(import).ok()?;
        let item = match ctx.scope.speculative_resolve(&path)? {
            hir::PathResolution::Macro(mac) => mac.into(),
            hir::PathResolution::Def(def) => def.into(),
            _ => return None,
        };
        let path = ctx.scope.module()?.find_use_path_prefixed(
            ctx.db,
            item,
            ctx.config.insert_use.prefix_kind,
        )?;
        Some((path.len() > 1).then(|| ImportEdit {
            import: LocatedImport::new(path.clone(), item, item, None),
            scope: import_scope.clone(),
        }))
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
) -> Option<(String, Option<String>)> {
    // validate that these are indeed simple paths
    // we can't save the paths unfortunately due to them not being Send+Sync
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
    let description = if description.is_empty() { None } else { Some(description.to_owned()) };
    Some((snippet, description))
}
