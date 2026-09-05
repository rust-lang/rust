//@ proc-macro: same-span-deref.rs

extern crate same_span_deref;

fn needs_sized<T: Sized>(_: T) {}

fn main() {
    let s: &str = "hello";
    needs_sized(same_span_deref::same_span_deref!(s));
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
}
