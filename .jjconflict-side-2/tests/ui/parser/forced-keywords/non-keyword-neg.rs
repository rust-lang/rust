// You are intentionally allowed to force identifiers to be treated as keywords even if they aren't
// actually reserved. We must ensure that we never treat them as normal identifiers when parsing.
//
//@ edition: 2021..
#![feature(forced_keywords)]

fn scope() {
    k#not_a_keyword()
    //~^ ERROR expected expression, found `k#not_a_keyword`
}
fn k#trying_to_pass_as_a_normal_ident() {}
//~^ ERROR expected identifier, found `k#trying_to_pass_as_a_normal_ident`
