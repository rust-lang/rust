//@ run-pass

fn foo(_: impl Into<f32>) {}

fn main() {
    foo(1.0);
}
