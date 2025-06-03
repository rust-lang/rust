//! Pretty printing of macros output.

use base_db::Crate;
use rustc_hash::FxHashMap;
use syntax::NodeOrToken;
use syntax::{SyntaxNode, ast::make};

use crate::{db::ExpandDatabase, span_map::ExpansionSpanMap};

/// Inserts whitespace and replaces `$crate` in macro expansions.
#[expect(deprecated)]
pub fn prettify_macro_expansion(
    db: &dyn ExpandDatabase,
    syn: SyntaxNode,
    span_map: &ExpansionSpanMap,
    target_crate_id: Crate,
) -> SyntaxNode {
    // Because `syntax_bridge::prettify_macro_expansion::prettify_macro_expansion()` clones subtree for `syn`,
    // that means it will be offsetted to the beginning.
    let span_offset = syn.text_range().start();
    let target_crate = target_crate_id.data(db);
    let mut syntax_ctx_id_to_dollar_crate_replacement = FxHashMap::default();
    syntax_bridge::prettify_macro_expansion::prettify_macro_expansion(
        syn,
        &mut |dollar_crate| {
            let ctx = span_map.span_at(dollar_crate.text_range().start() + span_offset).ctx;
            let replacement =
                syntax_ctx_id_to_dollar_crate_replacement.entry(ctx).or_insert_with(|| {
                    let macro_call_id = ctx
                        .outer_expn(db)
                        .expect("`$crate` cannot come from `SyntaxContextId::ROOT`");
                    let macro_call = db.lookup_intern_macro_call(macro_call_id.into());
                    let macro_def_crate = macro_call.def.krate;
                    // First, if this is the same crate as the macro, nothing will work but `crate`.
                    // If not, if the target trait has the macro's crate as a dependency, using the dependency name
                    // will work in inserted code and match the user's expectation.
                    // If not, the crate's display name is what the dependency name is likely to be once such dependency
                    // is inserted, and also understandable to the user.
                    // Lastly, if nothing else found, resort to leaving `$crate`.
                    if target_crate_id == macro_def_crate {
                        make::tokens::crate_kw()
                    } else if let Some(dep) =
                        target_crate.dependencies.iter().find(|dep| dep.crate_id == macro_def_crate)
                    {
                        make::tokens::ident(dep.name.as_str())
                    } else if let Some(crate_name) = &macro_def_crate.extra_data(db).display_name {
                        make::tokens::ident(crate_name.crate_name().as_str())
                    } else {
                        return dollar_crate.clone();
                    }
                });
            if replacement.text() == "$crate" {
                // The parent may have many children, and looking for the token may yield incorrect results.
                return None;
            }
            // We need to `clone_subtree()` but rowan doesn't provide such operation for tokens.
            let parent = replacement.parent().unwrap().clone_subtree().clone_for_update();
            parent
                .children_with_tokens()
                .filter_map(NodeOrToken::into_token)
                .find(|it| it.kind() == replacement.kind())
        },
        |_| (),
    )
}
