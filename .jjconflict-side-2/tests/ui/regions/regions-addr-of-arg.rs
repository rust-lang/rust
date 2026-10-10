// Check that taking the address of an argument yields a lifetime
// bounded by the current function call.

fn foo(a: isize) {
    let _p: &'static isize = &a; //~ ERROR `a` does not live long enough
}

fn bar(a: isize) {
    let _q: &isize = &a;
}

fn zed<'a>(a: isize) -> &'a isize {
    &a //~ ERROR cannot return reference to function parameter `a`
}

fn main() {
}
