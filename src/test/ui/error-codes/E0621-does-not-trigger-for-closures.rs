// Test that we give the generic error when one of the free regions is
// bound in a closure (rather than suggesting a change to the signature
// of the closure, which is not specified in `foo` but rather in `invoke`).

fn invoke<'a, F>(x: &'a i32, f: F) -> &'a i32
where F: FnOnce(&'a i32, &i32) -> &'a i32
{
    let y = 22;
    f(x, &y)
}

fn foo<'a>(x: &'a i32) {
    invoke(&x, |a, b| if a > b { a } else { b }); //~ ERROR lifetime may not live long enough
}

fn main() {}
