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
    str::from::utf8();
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP if there were a trait named `Example` with associated type `from`
    str::from::utf8_mut();
    //~^ ERROR ambiguous associated type [E0223]
    //~| HELP if there were a trait named `Example` with associated type `from`
}
