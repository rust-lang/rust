#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn parse1() {
    || use {
        //~^ ERROR expected one of `async`, `|`, or `||`, found `{`
    };
}

fn parse2() {
    move use || {
        //~^ ERROR expected one of `async`, `|`, or `||`, found keyword `use`
    };
}

fn parse3() {
    use move || {
        //~^ ERROR expected one of `async`, `|`, or `||`, found keyword `move`
    };
}

fn main() {}
