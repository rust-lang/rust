// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro]
pub fn cond(input: TokenStream) -> TokenStream {
    let mut conds = Vec::new();
    let mut input = input.into_iter().peekable();
    while let Some(tree) = input.next() {
        let cond = match tree {
            TokenTree::Group(tt) => tt.stream(),
            _ => panic!("Invalid input"),
        };
        let mut cond_trees = cond.clone().into_iter();
        let test = cond_trees.next().expect("Unexpected empty condition in `cond!`");
        let rhs = cond_trees.collect::<TokenStream>();
        if rhs.is_empty() {
            panic!("Invalid macro usage in cond: {}", cond);
        }
        let is_else = match test {
            TokenTree::Ident(ref word) => &*word.to_string() == "else",
            _ => false,
        };
        conds.push(if is_else || input.peek().is_none() {
            quote!({ $rhs })
        } else {
            quote!(if $test { $rhs } else)
        });
    }

    conds.into_iter().flat_map(|x| x.into_iter()).collect()
}
