fn main() {
    match ~"foo" {
        ['f', 'o', ..] => { } //~ ERROR mismatched types: expected `~str` but found a vector pattern
        _ => { }
    }
}
