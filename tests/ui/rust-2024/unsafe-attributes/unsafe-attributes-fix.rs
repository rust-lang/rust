//@ run-rustfix
#![deny(unsafe_attr_outside_unsafe)]

macro_rules! tt {
    ($e:tt) => {
        #$e
        extern fn foo() {}
    }
}

macro_rules! ident {
    ($e:ident) => {
        #[$e]
        //~^ ERROR: unsafe attribute used without unsafe
        //~| WARN this is accepted in the current edition
        extern fn bar() {}
    }
}

macro_rules! ident2 {
    ($e:ident, $l:literal) => {
        #[$e = $l]
        //~^ ERROR: unsafe attribute used without unsafe
        //~| WARN this is accepted in the current edition
        extern fn bars() {}
    }
}

macro_rules! meta {
    ($m:meta) => {
        #[$m]
        extern fn baz() {}
    }
}

macro_rules! meta2 {
    ($m:meta) => {
        #[$m]
        extern fn baw() {}
    }
}

tt!([no_mangle]);
//~^ ERROR: unsafe attribute used without unsafe
//~| WARN this is accepted in the current edition
ident!(no_mangle);
meta!(no_mangle);
//~^ ERROR: unsafe attribute used without unsafe
//~| WARN this is accepted in the current edition
meta2!(export_name = "baw");
//~^ ERROR: unsafe attribute used without unsafe
//~| WARN this is accepted in the current edition
ident2!(export_name, "bars");

#[no_mangle]
//~^ ERROR: unsafe attribute used without unsafe
//~| WARN this is accepted in the current edition
extern "C" fn one() {}

fn main() {}
