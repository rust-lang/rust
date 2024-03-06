//@ run-rustfix
#![deny(unused_qualifications)]
#![allow(deprecated, dead_code)]

mod foo {
    pub fn bar() {}
}

fn main() {
    use foo::bar;
    foo::bar(); //~ ERROR: unnecessary qualification
    crate::foo::bar(); //~ ERROR: unnecessary qualification
    bar();

    let _ = || -> Result<(), ()> { try!(Ok(())); Ok(()) }; // issue #37345

    let _ = std::string::String::new(); //~ ERROR: unnecessary qualification
    let _ = ::std::env::current_dir(); //~ ERROR: unnecessary qualification

    let _: std::vec::Vec<String> = std::vec::Vec::<String>::new();
    //~^ ERROR: unnecessary qualification
    //~| ERROR: unnecessary qualification

    use std::fmt;
    let _: std::fmt::Result = Ok(()); //~ ERROR: unnecessary qualification

    let _ = <bool as ::std::default::Default>::default(); // issue #121999
    //~^ ERROR: unnecessary qualification

    macro_rules! m { ($a:ident, $b:ident) => {
        $crate::foo::bar(); // issue #37357
        ::foo::bar(); // issue #38682
        foo::bar();
        foo::$b(); // issue #96698
        $a::bar();
    } }
    m!(foo, bar);
}

mod conflicting_names {
    mod std {}
    mod cell {}

    fn f() {
        let _ = ::std::env::current_dir();
        let _ = core::cell::Cell::new(1);
    }
}
