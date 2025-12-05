// issue #513

fn f() { }

fn main() {

    // f is not a bool
    if f { } //~ ERROR mismatched types
}
