struct Foo<'a>(&'a u8);
struct Baz<'a>(&'_ &'a u8); //~ ERROR missing lifetime specifier

fn foo<'_> //~ ERROR cannot be used here
(_: Foo<'_>) {}

trait Meh<'a> {}
impl<'a> Meh<'a> for u8 {}

fn meh() -> Box<dyn for<'_> Meh<'_>> //~ ERROR cannot be used here
//~^ ERROR missing lifetime specifier
{
  Box::new(5u8)
}

fn foo2(_: &'_ u8, y: &'_ u8) -> &'_ u8 { y } //~ ERROR missing lifetime specifier

fn main() {
    let x = 5;
    foo(Foo(&x));
    let _ = meh();
}
