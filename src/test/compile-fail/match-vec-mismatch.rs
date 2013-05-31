fn main() {
    match ~"foo" {
        ['f', 'o', .._] => { } //~ ERROR mismatched types: expected `~str` but found a vector pattern
        _ => { }
    }
}
