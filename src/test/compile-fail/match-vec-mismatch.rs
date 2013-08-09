fn main() {
    match ~"foo" {
        ['f', 'o', .._] => { } //~ ERROR mismatched types: expected `~str`, found a vector pattern
        _ => { }
    }
}
