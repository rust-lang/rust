fn main() {
    match ~"foo" {
        ['f', 'o', .._] => { } //~ ERROR mismatched type: expected `~str` but found vector
        _ => { }
    }
}
