struct Thing<'a>(&'a ());
struct Foo<'a>(&usize);
//~^ ERROR missing lifetime specifier

fn func1<'a>(_arg: &'a Thing) -> &() { unimplemented!() }
//~^ ERROR missing lifetime specifier
fn func2<'a>(_arg: &Thing<'a>) -> &() { unimplemented!() }
//~^ ERROR missing lifetime specifier

fn main() {}
