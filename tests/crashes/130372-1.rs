//@ known-bug: rust-lang/rust#130372

pub fn variadic_fn(n: usize, mut args: ...) {}

reuse variadic_fn;

fn main() {
        variadic_fn();
}
