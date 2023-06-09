// run-rustfix

#![allow(warnings)]

// This test checks that the following error is emitted when a `=` character is used to initialize
// a struct field when a `:` is expected.
//
// ```
// error: struct fields are initialized with a colon
//   --> $DIR/issue-57684.rs:12:20
//    |
// LL |     let _ = X { f1 = 5 };
//    |                    ^ help: replace equals symbol with a colon: `:`
// ```

struct X {
    f1: i32,
}

struct Y {
    f1: i32,
    f2: i32,
    f3: i32,
}

fn main() {
    let _ = X { f1 = 5 };
    //~^ ERROR expected `:`, found `=`

    let f3 = 3;
    let _ = Y {
        f1 = 5,
        //~^ ERROR expected `:`, found `=`
        f2: 4,
        f3,
    };
}
