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
        Some(Snippet {
            scope,
            label,
            snippet,
            description,
            requires: requires.iter().cloned().collect(), // Box::into doesn't work as that has a Copy bound 😒
        })
    }

    // FIXME: This shouldn't be fallible
    pub(crate) fn imports(
        &self,
        ctx: &CompletionContext,
        import_scope: &ImportScope,
    ) -> Result<Vec<ImportEdit>, ()> {
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
            scope,
            label,
            snippet,
            description,
            requires: requires.iter().cloned().collect(), // Box::into doesn't work as that has a Copy bound 😒
        })
    }

    // FIXME: This shouldn't be fallible
    pub(crate) fn imports(
        &self,
        ctx: &CompletionContext,
        import_scope: &ImportScope,
    ) -> Result<Vec<ImportEdit>, ()> {
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
) -> Result<Vec<ImportEdit>, ()> {
    let resolve = |import| {
        let path = ast::Path::parse(import).ok()?;
        match ctx.scope.speculative_resolve(&path)? {
            hir::PathResolution::Macro(_) => None,
            hir::PathResolution::Def(def) => {
                let item = def.into();
                let path = ctx.scope.module()?.find_use_path_prefixed(
                    ctx.db,
                    item,
                    ctx.config.insert_use.prefix_kind,
                )?;
                Some((path.len() > 1).then(|| ImportEdit {
                    import: LocatedImport::new(path.clone(), item, item, None),
                    scope: import_scope.clone(),
                }))
            }
            _ => None,
        }
    };
    let mut res = Vec::with_capacity(requires.len());
    for import in requires {
        match resolve(import) {
            Some(first) => res.extend(first),
            None => return Err(()),
        }
    }
    Ok(res)
}
