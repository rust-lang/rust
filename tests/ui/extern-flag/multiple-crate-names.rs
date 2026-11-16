//@ aux-build: somedep.rs
//@ compile-flags: -Zunstable-options
//@ edition:2018
//@ revisions: bare_first bar_first foo_first

cfg_select! {
    bare_first => {
        extern crate somedep;
        extern crate somedep as bar;
        extern crate somedep as foo;
    }
    bar_first => {
        extern crate somedep as bar;
        extern crate somedep as foo;
        extern crate somedep;
    }
    foo_first => {
        extern crate somedep as foo;
        extern crate somedep as bar;
        extern crate somedep;
    }
}
// use foo;

fn main() {
    // Excercise the path printing machinery to show that whichever way the crate is *first* named
    // with will be used as the crate name for paths. It will be `somedep::S`, `bar::S` or `foo::S`,
    // depending on which `extern` item is first.
    let _: S = bar::S; //~ ERROR: mismatched types
    let _: S = foo::S; //~ ERROR: mismatched types
    let _: bar::S = foo::S; // ok, they are the *same* type

    // Ensure that both crate names are findable and suggestable:
    fo::somefun(); //~ ERROR: cannot find module or crate `fo`
    ba::somefun(); //~ ERROR: cannot find module or crate `ba`
}

struct S;
