// no-reformat
// Testing various forms of `do` and `for` with empty arg lists

fn f(f: fn() -> bool) {
}

fn main() {
    do f() || { true }
    do f() { true }
    do f || { true }
    do f { true }
    for f() || { }
    for f() { }
    for f || { }
    for f { }
}
