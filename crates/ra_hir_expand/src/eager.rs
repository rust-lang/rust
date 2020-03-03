//! Eager expansion related utils
//!
//! Here is a dump of a discussion from Vadim Petrochenkov about Eager Expansion and
//! Its name resolution :
//!
//! > Eagerly expanded macros (and also macros eagerly expanded by eagerly expanded macros,
//! > which actually happens in practice too!) are resolved at the location of the "root" macro
//! > that performs the eager expansion on its arguments.
//! > If some name cannot be resolved at the eager expansion time it's considered unresolved,
//! > even if becomes available later (e.g. from a glob import or other macro).
//!
//! > Eagerly expanded macros don't add anything to the module structure of the crate and
//! > don't build any speculative module structures, i.e. they are expanded in a "flat"
//! > way even if tokens in them look like modules.
//!
//! > In other words, it kinda works for simple cases for which it was originally intended,
//! > and we need to live with it because it's available on stable and widely relied upon.
//!
//!
//! See the full discussion : https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler/topic/Eager.20expansion.20of.20built-in.20macros

use crate::{
    ast::{self, AstNode},
    db::AstDatabase,
    EagerCallLoc, EagerMacroId, InFile, MacroCallId, MacroCallKind, MacroDefId, MacroDefKind,
};

use ra_parser::FragmentKind;
use ra_syntax::{algo::replace_descendants, SyntaxElement, SyntaxNode};
use std::{collections::HashMap, sync::Arc};

pub fn expand_eager_macro(
    db: &impl AstDatabase,
    macro_call: InFile<ast::MacroCall>,
    def: MacroDefId,
    resolver: &dyn Fn(ast::Path) -> Option<MacroDefId>,
) -> Option<EagerMacroId> {
    let args = macro_call.value.token_tree()?;
    let parsed_args = mbe::ast_to_token_tree(&args)?.0;
    let parsed_args = mbe::token_tree_to_syntax_node(&parsed_args, FragmentKind::Expr).ok()?.0;
    let result = eager_macro_recur(db, macro_call.with_value(parsed_args.syntax_node()), resolver)?;

    let subtree = to_subtree(&result)?;

    if let MacroDefKind::BuiltInEager(eager) = def.kind {
        let (subtree, fragment) = eager.expand(&subtree).ok()?;
        let eager =
            EagerCallLoc { def, fragment, subtree: Arc::new(subtree), file_id: macro_call.file_id };

        Some(db.intern_eager_expansion(eager))
    } else {
        None
    }
}

fn to_subtree(node: &SyntaxNode) -> Option<tt::Subtree> {
    let mut subtree = mbe::syntax_node_to_token_tree(node)?.0;
    subtree.delimiter = None;
    Some(subtree)
}

fn lazy_expand(
    db: &impl AstDatabase,
    def: &MacroDefId,
    macro_call: InFile<ast::MacroCall>,
) -> Option<InFile<SyntaxNode>> {
    let ast_id = db.ast_id_map(macro_call.file_id).ast_id(&macro_call.value);

    let id: MacroCallId =
        def.as_lazy_macro(db, MacroCallKind::FnLike(macro_call.with_value(ast_id))).into();

    db.parse_or_expand(id.as_file()).map(|node| InFile::new(id.as_file(), node))
}

fn eager_macro_recur(
    db: &impl AstDatabase,
    curr: InFile<SyntaxNode>,
    macro_resolver: &dyn Fn(ast::Path) -> Option<MacroDefId>,
) -> Option<SyntaxNode> {
    let mut original = curr.value.clone();

    let children = curr.value.descendants().filter_map(ast::MacroCall::cast);
    let mut replaces: HashMap<SyntaxElement, SyntaxElement> = HashMap::default();

    // Collect replacement
    for child in children {
        let def: MacroDefId = macro_resolver(child.path()?)?;
        let insert = match def.kind {
            MacroDefKind::BuiltInEager(_) => {
                let id: MacroCallId =
                    expand_eager_macro(db, curr.with_value(child.clone()), def, macro_resolver)?
                        .into();
                db.parse_or_expand(id.as_file())?
            }
            MacroDefKind::Declarative
            | MacroDefKind::BuiltIn(_)
            | MacroDefKind::BuiltInDerive(_) => {
                let expanded = lazy_expand(db, &def, curr.with_value(child.clone()))?;
                // replace macro inside
                eager_macro_recur(db, expanded, macro_resolver)?
            }
        };

        replaces.insert(child.syntax().clone().into(), insert.into());
    }

    if !replaces.is_empty() {
        original = replace_descendants(&original, |n| replaces.get(n).cloned());
    }

    Some(original)
}
