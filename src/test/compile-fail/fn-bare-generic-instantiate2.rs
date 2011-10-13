// error-pattern: generic bare functions can only be called or bound
// Issue #1038

fn main() {
    fn# foo<T>(i: T) { }

    // This wants to build a closure over type int,
    // but there's no way to do that while still being a bare function
    f(foo);
}

fn f(i: fn#(&&int)) {
}