// Test successful and unsuccessful parsing of the `default` contextual keyword

#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

trait Foo {
    fn foo<T: Default>() -> T;
}

impl Foo for u8 {
    default fn foo<T: Default>() -> T {
        T::default()
    }
}

impl Foo for u16 {
    pub default fn foo<T: Default>() -> T { //~ ERROR visibility qualifiers are not permitted here
        T::default()
    }
}

impl Foo for u32 { //~ ERROR not all trait items implemented, missing: `foo`
    default pub fn foo<T: Default>() -> T { T::default() }
    //~^ ERROR `default` is not followed by an item
    //~| ERROR non-item in item list
}

fn main() {}
