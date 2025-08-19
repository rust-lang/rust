//@ edition: 2015
//@ run-rustfix
#![deny(unused_qualifications)]
#![deny(unused_imports)]
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

    let _: std::vec::Vec<String> = std::vec::Vec::<String>::new();
    //~^ ERROR: unnecessary qualification
    //~| ERROR: unnecessary qualification

    use std::fmt;
    //~^ ERROR: unused import: `std::fmt`
    let _: std::fmt::Result = Ok(());
    // don't report unnecessary qualification because fix(#122373) for issue #121331

    let _ = <bool as std::default::Default>::default(); // issue #121999 (modified)
    //~^ ERROR: unnecessary qualification

    macro_rules! m { ($a:ident, $b:ident) => {
        $crate::foo::bar(); // issue #37357
        ::foo::bar(); // issue #38682
        foo::bar();
        foo::$b(); // issue #96698
        $a::bar();
        $a::$b();
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
