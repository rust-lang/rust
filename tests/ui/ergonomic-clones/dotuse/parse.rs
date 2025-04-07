#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn parse1() {
    1.use!;
    //~^ ERROR: expected one of `.`, `;`, `?`, `}`, or an operator, found `!`
    }

fn parse2() {
    1.use!(2);
    //~^ ERROR: expected one of `.`, `;`, `?`, `}`, or an operator, found `!`
    }

fn parse3() {
    1.use 2;
    //~^ ERROR: expected one of `.`, `;`, `?`, `}`, or an operator, found `2`
    }

fn parse4() {
    1.use? 2;
    //~^ ERROR: expected one of `.`, `;`, `?`, `}`, or an operator, found `2`
    }

fn parse5() {
    1.use();
    //~^ ERROR: incorrect use of `use`
}

fn parse6() {
    1.use(2);
    //~^ ERROR: expected function, found `{integer}` [E0618]
}

fn parse7() {
    1.use { 2 };
    //~^ ERROR: expected one of `.`, `;`, `?`, `}`, or an operator, found `{`
}

fn main() {}
