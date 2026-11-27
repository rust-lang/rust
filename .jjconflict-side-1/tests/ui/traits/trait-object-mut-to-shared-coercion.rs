//! Tests that coercion from `&mut dyn Trait` to `&dyn Trait` works correctly.

//@ run-pass

trait Foo {
    fn foo(&self) -> usize;
    fn bar(&mut self) -> usize;
}

impl Foo for usize {
    fn foo(&self) -> usize {
        *self
    }

    fn bar(&mut self) -> usize {
        *self += 1;
        *self
    }
}

fn do_it_mut(obj: &mut dyn Foo) {
    let x = obj.bar();
    let y = obj.foo();
    assert_eq!(x, y);

    do_it_imm(obj, y);
}

fn do_it_imm(obj: &dyn Foo, v: usize) {
    let y = obj.foo();
    assert_eq!(v, y);
}

pub fn main() {
    let mut x: usize = 22;
    let obj = &mut x as &mut dyn Foo;
    do_it_mut(obj);
    do_it_imm(obj, 23);
    do_it_mut(obj);
}
