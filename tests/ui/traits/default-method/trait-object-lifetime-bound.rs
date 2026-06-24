//@ run-pass
trait Foo<'a> {
    fn bar<'b>(&self, x: &'b u8) -> u8 where 'a: 'b { *x+7 }
}

pub struct FooBar;
impl Foo<'static> for FooBar {}
fn test(foobar: FooBar) -> Box<dyn Foo<'static>> {
    Box::new(foobar)
}

fn main() {
    assert_eq!(test(FooBar).bar(&4), 11);
}
