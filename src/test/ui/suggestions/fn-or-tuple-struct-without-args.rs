fn foo(a: usize, b: usize) -> usize { a }

fn bar() -> usize { 42 }

struct S(usize, usize);
struct V();

trait T {
    fn baz(x: usize, y: usize) -> usize { x }
    fn bat() -> usize { 42 }
}

fn main() {
    let _: usize = foo; //~ ERROR mismatched types
    let _: S = S; //~ ERROR mismatched types
    let _: usize = bar; //~ ERROR mismatched types
    let _: V = V; //~ ERROR mismatched types
    let _: usize = T::baz; //~ ERROR mismatched types
    let _: usize = T::bat; //~ ERROR mismatched types
}
