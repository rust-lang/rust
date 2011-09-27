// error-pattern: mismatched kind

resource r(b: bool) {
}

fn main() {
    // Kind analysis considers this a copy, which isn't strictly true,
    // but for many assignment initializers could be.  To actually
    // assign a resource to a local we can still use a move
    // initializer.
    let i = r(true);
}