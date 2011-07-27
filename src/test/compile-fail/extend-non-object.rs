//error-pattern:x does not have object type
use std;

fn main() {
    let x = 3;

    let anon_obj =
        obj () {
            fn foo() -> int { ret 3; }
            with
            x
        };
}