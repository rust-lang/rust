#![feature(generic_associated_types)]
//~^ WARNING: the feature `generic_associated_types` is incomplete

trait X {
    type Y<'a>;
}

fn f1<'a>(arg : Box<dyn X<Y = B = &'a ()>>) {}
    //~^ ERROR: expected one of `!`, `(`, `+`, `,`, `::`, `<`, or `>`, found `=`

fn main() {}
