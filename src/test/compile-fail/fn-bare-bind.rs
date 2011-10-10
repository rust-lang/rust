// error-pattern:mismatched types: expected fn#() but found fn()

fn# f() {
}

fn main() {
    // Can't produce a bare function by binding
    let g: fn#() = bind f();
}