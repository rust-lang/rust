#![crate_type = "lib"]

#[align(16)]
//~^ ERROR the `#[align]` attribute is an experimental feature
fn requires_alignment() {}

trait MyTrait {
    #[align]
    //~^ ERROR the `#[align]` attribute is an experimental feature
    //~| ERROR malformed `align` attribute input
    fn myfun();
}
