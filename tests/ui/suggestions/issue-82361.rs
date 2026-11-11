//@ run-rustfix

fn main() {
    let a: usize = 123;
    let b: &usize = &a;

    if true {
        a
    } else {
        b //~ ERROR `if` and `else` have incompatible types [E0308]
    };

    if true {
        1
    } else {
        &1 //~ ERROR `if` and `else` have incompatible types [E0308]
    };

    if true {
        1
    } else {
        &mut 1 //~ ERROR `if` and `else` have incompatible types [E0308]
    };
}
