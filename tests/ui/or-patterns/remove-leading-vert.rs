// Test the suggestion to remove a leading, or trailing `|`.

//@ run-rustfix

#![allow(warnings)]

fn main() {}

#[cfg(FALSE)]
fn leading() {
    fn fun1( | A: E) {} //~ ERROR top-level or-patterns are not allowed
    fn fun2( || A: E) {} //~ ERROR unexpected `||` before function parameter
    let ( | A): E;
    let ( || A): (E); //~ ERROR unexpected token `||` in pattern
    let ( | A,): (E,);
    let [ | A ]: [E; 1];
    let [ || A ]: [E; 1]; //~ ERROR unexpected token `||` in pattern
    let TS( | A ): TS;
    let TS( || A ): TS; //~ ERROR unexpected token `||` in pattern
    let NS { f: | A }: NS;
    let NS { f: || A }: NS; //~ ERROR unexpected token `||` in pattern
}

#[cfg(FALSE)]
fn trailing() {
    let ( A | ): E; //~ ERROR a trailing `|` is not allowed in an or-pattern
    let (a |,): (E,); //~ ERROR a trailing `|` is not allowed in an or-pattern
    let ( A | B | ): E; //~ ERROR a trailing `|` is not allowed in an or-pattern
    let [ A | B | ]: [E; 1]; //~ ERROR a trailing `|` is not allowed in an or-pattern
    let S { f: B | }; //~ ERROR a trailing `|` is not allowed in an or-pattern
    let ( A || B | ): E; //~ ERROR unexpected token `||` in pattern
    //~^ ERROR a trailing `|` is not allowed in an or-pattern
    match A {
        A | => {} //~ ERROR a trailing `|` is not allowed in an or-pattern
        A || => {} //~ ERROR a trailing `|` is not allowed in an or-pattern
        A || B | => {} //~ ERROR unexpected token `||` in pattern
        //~^ ERROR a trailing `|` is not allowed in an or-pattern
        | A | B | => {}
        //~^ ERROR a trailing `|` is not allowed in an or-pattern
    }

    // These test trailing-vert in `let` bindings, but they also test that we don't emit a
    // duplicate suggestion that would confuse rustfix.

    let a | : u8 = 0; //~ ERROR a trailing `|` is not allowed in an or-pattern
    let a | = 0; //~ ERROR a trailing `|` is not allowed in an or-pattern
    let a | ; //~ ERROR a trailing `|` is not allowed in an or-pattern
}
