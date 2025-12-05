//@ aux-build:extern_macro_crate.rs
#[macro_use(myprintln, myprint)]
extern crate extern_macro_crate;

fn main() {
    myprintln!("{}");
    //~^ ERROR in format string
}
