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

use ra_db::CrateId;
use ra_parser::FragmentKind;
use ra_syntax::{algo::SyntaxRewriter, SyntaxNode};
use std::sync::Arc;

pub fn expand_eager_macro(
    db: &dyn AstDatabase,
    krate: CrateId,
    macro_call: InFile<ast::MacroCall>,
    def: MacroDefId,
    resolver: &dyn Fn(ast::Path) -> Option<MacroDefId>,
) -> Option<EagerMacroId> {
    let args = macro_call.value.token_tree()?;
    let parsed_args = mbe::ast_to_token_tree(&args)?.0;

    // Note:
    // When `lazy_expand` is called, its *parent* file must be already exists.
    // Here we store an eager macro id for the argument expanded subtree here
    // for that purpose.
    let arg_id = db.intern_eager_expansion({
        EagerCallLoc {
            def,
            fragment: FragmentKind::Expr,
            subtree: Arc::new(parsed_args.clone()),
            krate,
            file_id: macro_call.file_id,
        }
    });
    let arg_file_id: MacroCallId = arg_id.into();

    let parsed_args = mbe::token_tree_to_syntax_node(&parsed_args, FragmentKind::Expr).ok()?.0;
    let result = eager_macro_recur(
        db,
        InFile::new(arg_file_id.as_file(), parsed_args.syntax_node()),
        krate,
        resolver,
    )?;
    let subtree = to_subtree(&result)?;

    if let MacroDefKind::BuiltInEager(eager) = def.kind {
        let (subtree, fragment) = eager.expand(db, arg_id, &subtree).ok()?;
        let eager = EagerCallLoc {
            def,
            fragment,
            subtree: Arc::new(subtree),
            krate,
            file_id: macro_call.file_id,
        };

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
    db: &dyn AstDatabase,
    def: &MacroDefId,
    macro_call: InFile<ast::MacroCall>,
    krate: CrateId,
) -> Option<InFile<SyntaxNode>> {
    let ast_id = db.ast_id_map(macro_call.file_id).ast_id(&macro_call.value);

    let id: MacroCallId =
        def.as_lazy_macro(db, krate, MacroCallKind::FnLike(macro_call.with_value(ast_id))).into();

    db.parse_or_expand(id.as_file()).map(|node| InFile::new(id.as_file(), node))
}

fn eager_macro_recur(
    db: &dyn AstDatabase,
    curr: InFile<SyntaxNode>,
    krate: CrateId,
    macro_resolver: &dyn Fn(ast::Path) -> Option<MacroDefId>,
) -> Option<SyntaxNode> {
    let original = curr.value.clone();

    let children = curr.value.descendants().filter_map(ast::MacroCall::cast);
    let mut rewriter = SyntaxRewriter::default();

    // Collect replacement
    for child in children {
        let def: MacroDefId = macro_resolver(child.path()?)?;
        let insert = match def.kind {
            MacroDefKind::BuiltInEager(_) => {
                let id: MacroCallId = expand_eager_macro(
                    db,
                    krate,
                    curr.with_value(child.clone()),
                    def,
                    macro_resolver,
                )?
                .into();
                db.parse_or_expand(id.as_file())?
            }
            MacroDefKind::Declarative
            | MacroDefKind::BuiltIn(_)
            | MacroDefKind::BuiltInDerive(_)
            | MacroDefKind::CustomDerive(_) => {
                let expanded = lazy_expand(db, &def, curr.with_value(child.clone()), krate)?;
                // replace macro inside
                eager_macro_recur(db, expanded, krate, macro_resolver)?
            }
        };

        rewriter.replace(child.syntax(), &insert);
    }

    let res = rewriter.rewrite(&original);
    Some(res)
}
