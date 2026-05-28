//@ check-pass

trait T<'a> {}

fn foo(_: Vec<Box<dyn T<'_>>>) {}

fn main() {
    foo(vec![]);
}
