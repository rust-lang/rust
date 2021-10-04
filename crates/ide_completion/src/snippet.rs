//! User (postfix)-snippet definitions.
//!
//! Actual logic is implemented in [`crate::completions::postfix`] and [`crate::completions::snippet`].
use ide_db::helpers::{import_assets::LocatedImport, insert_use::ImportScope};
use itertools::Itertools;
use syntax::ast;

use crate::{context::CompletionContext, ImportEdit};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PostfixSnippetScope {
    Expr,
    Type,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnippetScope {
    Item,
    Expr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PostfixSnippet {
    pub scope: PostfixSnippetScope,
    pub label: String,
    snippet: String,
    pub description: Option<String>,
    pub requires: Box<[String]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Snippet {
    pub scope: SnippetScope,
    pub label: String,
    pub snippet: String,
    pub description: Option<String>,
    pub requires: Box<[String]>,
}
impl Snippet {
    pub fn new(
        label: String,
        snippet: &[String],
        description: &[String],
        requires: &[String],
        scope: SnippetScope,
    ) -> Option<Self> {
        let (snippet, description) = validate_snippet(snippet, description, requires)?;
        Some(Snippet {
            scope,
            label,
            snippet,
            description,
            requires: requires.iter().cloned().collect(), // Box::into doesn't work as that has a Copy bound 😒
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

    pub fn is_item(&self) -> bool {
        self.scope == SnippetScope::Item
    }

    pub fn is_expr(&self) -> bool {
        self.scope == SnippetScope::Expr
    }
}

impl PostfixSnippet {
    pub fn new(
        label: String,
        snippet: &[String],
        description: &[String],
        requires: &[String],
        scope: PostfixSnippetScope,
    ) -> Option<Self> {
        let (snippet, description) = validate_snippet(snippet, description, requires)?;
        Some(PostfixSnippet {
            scope,
            label,
            snippet,
            description,
            requires: requires.iter().cloned().collect(), // Box::into doesn't work as that has a Copy bound 😒
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

    pub fn snippet(&self, receiver: &str) -> String {
        self.snippet.replace("$receiver", receiver)
    }

    pub fn is_item(&self) -> bool {
        self.scope == PostfixSnippetScope::Type
    }

    pub fn is_expr(&self) -> bool {
        self.scope == PostfixSnippetScope::Expr
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
    description: &[String],
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
    let description = description.iter().join("\n");
    let description = if description.is_empty() { None } else { Some(description) };
    Some((snippet, description))
}
