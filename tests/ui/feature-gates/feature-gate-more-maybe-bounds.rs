#![feature(auto_traits, lang_items)]

#[lang = "default_trait1"] trait Trait1 {}
#[lang = "default_trait2"] auto trait Trait2 {}

trait Trait3: ?Trait1 {}
//~^ ERROR relaxed bounds are not permitted in supertrait bounds
//~| ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR bound modifier `?` can only be applied to `Sized`

fn foo(_: Box<dyn Trait1 + ?Trait2>) {}
//~^  ERROR relaxed bounds are not permitted in trait object types
//~| ERROR bound modifier `?` can only be applied to `Sized`
fn bar<T: ?Trait1 + ?Trait2>(_: T) {}
//~^ ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR bound modifier `?` can only be applied to `Sized`

fn main() {}
