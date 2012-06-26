pure fn f(f: fn()) {
}

pure fn g() {
    // `f { || }` is considered pure, so `do f { || }` should be too
    do f { || }
}

fn main() {
}