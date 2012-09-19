// error-pattern:mismatched types: expected `()` but found `bool`

fn main() {
    for vec::each_ref(~[0]) |_i| {
        true
    }
}