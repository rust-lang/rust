//@ run-rustfix
#![allow(unused_variables, dead_code)]
struct Struct;
struct Struct2;
// We use a second one here because otherwise when applying suggestions we'd end up with two
// `#[derive(Clone)]` annotations.

fn test1() {
    let mut val = Some(Struct);
    while let Some(foo) = val { //~ ERROR use of moved value
        if true {
            val = None;
        } else {

        }
    }
}

fn test2() {
    let mut foo = Some(Struct);
    let _x = foo.unwrap();
    if true {
        foo = Some(Struct);
    } else {
    }
    let _y = foo; //~ ERROR use of moved value: `foo`
}

fn test3() {
    let mut foo = Some(Struct2);
    let _x = foo.unwrap();
    if true {
        foo = Some(Struct2);
    } else if true {
        foo = Some(Struct2);
    } else if true {
        foo = Some(Struct2);
    } else if true {
        foo = Some(Struct2);
    } else {
    }
    let _y = foo; //~ ERROR use of moved value: `foo`
}

fn main() {}
