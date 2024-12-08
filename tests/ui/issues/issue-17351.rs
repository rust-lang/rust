//@ run-pass

trait Str { fn foo(&self) {} } //~ WARN method `foo` is never used
impl Str for str {}
impl<'a, S: ?Sized> Str for &'a S where S: Str {}

fn main() {
    let _: &dyn Str = &"x";
}
