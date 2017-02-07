fn foo() {}
fn bar(u: u32) { assert_eq!(u, 42); }
fn baa(u: u32, f: f32) {
    assert_eq!(u, 42);
    assert_eq!(f, 3.141);
}

fn main() {
    let f: &Fn() = &(foo as fn());
    f();
    let f: &Fn(u32) = &(bar as fn(u32));
    f(42);
    let f: &Fn(u32, f32) = &(baa as fn(u32, f32));
    f(42, 3.141);
}
