//@ run-pass

#![allow(incomplete_features, unused_variables)]
#![feature(move_expr)]

fn main() {
    let c = || move({
        let x = 3;
        x
    });
    assert_eq!(c(), 3);

    let c = || move({
        let x = 4;
        (|| x)()
    });
    assert_eq!(c(), 4);

    let outer = || {
        let x = 5;
        || move(x)
    };
    assert_eq!(outer()(), 5);

    let outer = || {
        let x = 6;
        || move({
            let y = 7;
            || move(y + x)
        })
    };
    assert_eq!(outer()()(), 13);

    let c = || move({
        fn identity(x: i32) -> i32 {
            x
        }
        identity(8)
    });
    assert_eq!(c(), 8);

    let c = || move(const { 9 });
    assert_eq!(c(), 9);

    let x = 10;
    let c = || move(x);
    let d = || move(x + 1);
    assert_eq!(c(), 10);
    assert_eq!(d(), 11);

    let c = || {
        let x = 12;
        move({
            let x = 13;
            x
        })
    };
    assert_eq!(c(), 13);

    let c = |x: i32| move({
        let x;
        x = 14;
        x
    });
    assert_eq!(c(0), 14);
}
