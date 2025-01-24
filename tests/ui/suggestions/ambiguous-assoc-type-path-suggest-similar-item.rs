// https://github.com/rust-lang/rust/issues/109195
struct Foo;

impl Foo {
    fn bar_baz() {}
}

impl Foo {
    fn bar_quux() {}
}

fn main() {
    String::from::utf8;
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP there is an associated function with a similar name: `from_utf8`
    String::from::utf8();
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP there is an associated function with a similar name: `from_utf8`
    String::from::utf16();
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP there is an associated function with a similar name: `from_utf16`
    String::from::method_that_doesnt_exist();
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP if there were a trait named `Example` with associated type `from`
    str::into::string();
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP there is an associated function with a similar name: `into_string`
    str::char::indices();
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP there is an associated function with a similar name: `char_indices`
    Foo::bar::baz;
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP there is an associated function with a similar name: `bar_baz`
    Foo::bar::quux;
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP there is an associated function with a similar name: `bar_quux`
    Foo::bar::fizz;
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP if there were a trait named `Example` with associated type `bar`
    i32::wrapping::add;
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP there is an associated function with a similar name: `wrapping_add`
    i32::wrapping::method_that_doesnt_exist;
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP if there were a trait named `Example` with associated type `wrapping`

    // this one ideally should suggest `downcast_mut_unchecked`
    <dyn std::any::Any>::downcast::mut_unchecked;
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP if there were a trait named `Example` with associated type `downcast`
}
