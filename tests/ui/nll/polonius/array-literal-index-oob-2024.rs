// This test used to ICE under `-Zpolonius=next` when computing loan liveness
// and taking kills into account during reachability traversal of the localized
// constraint graph. Originally from another test but on edition 2024, as
// seen in issue #135646.

//@ compile-flags: -Zpolonius=next
//@ edition: 2024
//@ check-pass

fn main() {
    &{ [1, 2, 3][4] };
}
