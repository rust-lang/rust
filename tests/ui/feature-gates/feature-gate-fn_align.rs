#![crate_type = "lib"]

// ignore-tidy-linelength

// FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity

#[rustc_align(16)]
//~^ ERROR the `#[rustc_align]` attribute is an experimental feature
fn requires_alignment() {}

trait MyTrait {
    #[rustc_align]
    //~^ ERROR the `#[rustc_align]` attribute is an experimental feature
    //~| ERROR malformed `rustc_align` attribute input
    fn myfun();
}
