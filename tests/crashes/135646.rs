//@ known-bug: #135646
//@ compile-flags: --edition=2024 -Zpolonius=next
fn main() {
    &{ [1, 2, 3][4] };
}
