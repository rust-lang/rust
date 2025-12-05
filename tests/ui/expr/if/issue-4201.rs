//@ dont-require-annotations: NOTE

fn main() {
    let a = if true {
        0
    } else if false {
//~^ ERROR `if` may be missing an `else` clause
//~| NOTE expected integer, found `()`
        1
    };
}
