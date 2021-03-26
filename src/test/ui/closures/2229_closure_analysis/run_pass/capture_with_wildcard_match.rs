//check-pass
#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

fn test1() {
    let foo : [Vec<u8>; 3] = ["String".into(), "String".into(), "String".into()];
    let c = || {
        match foo { _ => () };
    };
    drop(foo);
    c();
}

fn test2() {
    let foo : Option<[Vec<u8>; 3]> = Some(["String".into(), "String".into(), "String".into()]);
    let c = || {
        match foo {
            Some(_) => 1,
            _ => 2
        };
    };
    c();
}

fn main() {
    test1();
    test2();
}
