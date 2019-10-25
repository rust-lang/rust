// Test the suggestion to remove a leading `|`.

// run-rustfix

#![feature(or_patterns)]
#![allow(warnings)]

fn main() {}

#[cfg(FALSE)]
fn leading_vert() {
    fn fun1( | A: E) {} //~ ERROR a leading `|` is not allowed in a parameter pattern
    fn fun2( || A: E) {} //~ ERROR a leading `|` is not allowed in a parameter pattern
    let ( | A): E; //~ ERROR a leading `|` is only allowed in a top-level pattern
    let ( || A): (E); //~ ERROR a leading `|` is only allowed in a top-level pattern
    let ( | A,): (E,); //~ ERROR a leading `|` is only allowed in a top-level pattern
    let [ | A ]: [E; 1]; //~ ERROR a leading `|` is only allowed in a top-level pattern
    let [ || A ]: [E; 1]; //~ ERROR a leading `|` is only allowed in a top-level pattern
    let TS( | A ): TS; //~ ERROR a leading `|` is only allowed in a top-level pattern
    let TS( || A ): TS; //~ ERROR a leading `|` is only allowed in a top-level pattern
    let NS { f: | A }: NS; //~ ERROR a leading `|` is only allowed in a top-level pattern
    let NS { f: || A }: NS; //~ ERROR a leading `|` is only allowed in a top-level pattern
}
