// There's a suggestion that turns `Iterator<u32>` into `Iterator<Item = u32>`
// if we have more generics than the trait wants. Let's not consider RPITITs
// for this, since that makes no sense right now.

trait Foo {
    fn bar(self) -> impl Sized;
}

impl Foo<u8> for () {
    //~^ ERROR trait takes 0 generic arguments but 1 generic argument was supplied
    fn bar(self) -> impl Sized {}
}

fn main() {}
