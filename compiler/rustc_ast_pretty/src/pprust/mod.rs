#[cfg(test)]
mod tests;

pub mod state;
pub use state::{print_crate, AnnNode, Comments, PpAnn, PrintState, State};

use rustc_ast as ast;
use rustc_ast::token::{Nonterminal, Token, TokenKind};
use rustc_ast::tokenstream::{TokenStream, TokenTree};

use std::borrow::Cow;

pub fn nonterminal_to_string(nt: &Nonterminal) -> String {
    State::new().nonterminal_to_string(nt)
}

/// Print the token kind precisely, without converting `$crate` into its respective crate name.
pub fn token_kind_to_string(tok: &TokenKind) -> Cow<'static, str> {
    State::new().token_kind_to_string(tok)
}

/// Print the token precisely, without converting `$crate` into its respective crate name.
pub fn token_to_string(token: &Token) -> Cow<'static, str> {
    State::new().token_to_string(token)
}

pub fn ty_to_string(ty: &ast::Ty) -> String {
    State::new().ty_to_string(ty)
}

pub fn bounds_to_string(bounds: &[ast::GenericBound]) -> String {
    State::new().bounds_to_string(bounds)
}

pub fn where_bound_predicate_to_string(where_bound_predicate: &ast::WhereBoundPredicate) -> String {
    State::new().where_bound_predicate_to_string(where_bound_predicate)
}

pub fn pat_to_string(pat: &ast::Pat) -> String {
    State::new().pat_to_string(pat)
}

pub fn expr_to_string(e: &ast::Expr) -> String {
    State::new().expr_to_string(e)
}

pub fn tt_to_string(tt: &TokenTree) -> String {
    State::new().tt_to_string(tt)
}

pub fn tts_to_string(tokens: &TokenStream) -> String {
    State::new().tts_to_string(tokens)
}

pub fn item_to_string(i: &ast::Item) -> String {
    State::new().item_to_string(i)
}

pub fn path_to_string(p: &ast::Path) -> String {
    State::new().path_to_string(p)
}

pub fn path_segment_to_string(p: &ast::PathSegment) -> String {
    State::new().path_segment_to_string(p)
}

pub fn vis_to_string(v: &ast::Visibility) -> String {
    State::new().vis_to_string(v)
}

pub fn meta_list_item_to_string(li: &ast::NestedMetaItem) -> String {
    State::new().meta_list_item_to_string(li)
}

pub fn attribute_to_string(attr: &ast::Attribute) -> String {
    State::new().attribute_to_string(attr)
}

pub fn to_string(f: impl FnOnce(&mut State<'_>)) -> String {
    State::to_string(f)
}

pub fn crate_to_string_for_macros(krate: &ast::Crate) -> String {
    State::to_string(|s| {
        s.print_inner_attributes(&krate.attrs);
        for item in &krate.items {
            s.print_item(item);
        }
    })
}
