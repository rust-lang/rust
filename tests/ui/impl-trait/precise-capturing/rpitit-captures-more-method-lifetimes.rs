// Make sure we don't ICE when an RPITIT captures more method args than the
// trait definition, which is not allowed. Due to the default lifetime capture
// rules of RPITITs, this is only doable if we use precise capturing.

pub trait Foo {
    fn bar<'tr: 'tr>(&'tr mut self) -> impl Sized + use<Self>;
    //~^ ERROR `use<...>` precise capturing syntax is currently not allowed in return-position `impl Trait` in traits
}

impl Foo for () {
    fn bar<'im: 'im>(&'im mut self) -> impl Sized + 'im {}
    //~^ ERROR return type captures more lifetimes than trait definition
    //~| WARN impl trait in impl method signature does not match trait method signature
}

fn main() {}
