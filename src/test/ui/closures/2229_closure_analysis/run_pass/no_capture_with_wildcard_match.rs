//check-pass
#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

fn test1() {
    let foo = [1, 2, 3];
    let c = || {
        match foo { _ => () };
    };
}

fn test2() {
    let foo = Some([1, 2, 3]);
    let c = || {
        match foo {
            Some(_) => 1,
            _ => 2
        };
    };
}

fn main() {
    test1();
    test2();
}
