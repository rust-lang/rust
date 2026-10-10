//@ check-fail

trait Bar<T> {
    fn bar(&self, _: T) {}
}

trait Foo: Bar<i32> + Bar<u32> {
    fn foo(&self, _: ()) {}
}

struct S;

impl Bar<i32> for S {}
impl Bar<u32> for S {}
impl Foo for S {}

fn main() {
    let s: &dyn Foo = &S;
    let t: &dyn Bar<_> = s; //~ ERROR mismatched types
}
