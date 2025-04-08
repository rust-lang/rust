//@ known-bug: #135646
//@ compile-flags: -Zpolonius=next
//@ edition: 2024

fn main() {
    &{ [1, 2, 3][4] };
}
