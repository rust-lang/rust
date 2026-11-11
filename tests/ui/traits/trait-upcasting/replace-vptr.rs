//@ run-pass

trait A {
    fn foo_a(&self); //~ WARN method `foo_a` is never used
}

trait B {
    fn foo_b(&self);
}

trait C: A + B {
    fn foo_c(&self); //~ WARN method `foo_c` is never used
}

struct S(i32);

impl A for S {
    fn foo_a(&self) {
        unreachable!();
    }
}

impl B for S {
    fn foo_b(&self) {
        assert_eq!(42, self.0);
    }
}

impl C for S {
    fn foo_c(&self) {
        unreachable!();
    }
}

fn invoke_inner(b: &dyn B) {
    b.foo_b();
}

fn invoke_outer(c: &dyn C) {
    invoke_inner(c);
}

fn main() {
    let s = S(42);
    invoke_outer(&s);
}
