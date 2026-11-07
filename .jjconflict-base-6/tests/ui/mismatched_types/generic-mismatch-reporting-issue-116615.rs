fn foo<T>(a: T, b: T) {}
fn foo_multi_same<T>(a: T, b: T, c: T, d: T, e: T, f: i32) {}
fn foo_multi_generics<S, T>(a: T, b: T, c: T, d: T, e: T, f: S, g: S) {}

fn main() {
    foo(1, 2.);
    //~^  ERROR mismatched types
    foo_multi_same("a", "b", false, true, (), 32);
    //~^  ERROR arguments to this function are incorrect
    foo_multi_generics("a", "b", "c", true, false, 32, 2.);
    //~^  ERROR arguments to this function are incorrect
    foo_multi_same("a", 1, 2, "d", "e", 32);
    //~^  ERROR arguments to this function are incorrect
}
