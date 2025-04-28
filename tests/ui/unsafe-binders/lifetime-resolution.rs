#![feature(unsafe_binders)]
//~^ WARN the feature `unsafe_binders` is incomplete

fn foo<'a>() {
    let good: unsafe<'b> &'a &'b ();

    let missing: unsafe<> &'missing ();
    //~^ ERROR use of undeclared lifetime name `'missing`

    fn inner<'b>() {
        let outer: unsafe<> &'a &'b ();
        //~^ ERROR can't use generic parameters from outer item
    }
}

fn main() {}
