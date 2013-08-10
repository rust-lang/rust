fn a() {
    let x = ~[1];
    match x {
        [_, _, _, _, _, .._] => fail!(),
        [.._, _, _, _, _] => fail!(),
        [_, .._, _, _] => fail!(),
        [_, _] => fail!(),
        [a] => {
            assert_eq!(a, 1);
        }
        [] => fail!()
    }
}

fn b() {
    let x = ~[1, 2, 3];
    match x {
        [a, b, ..c] => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            assert_eq!(c, &[3]);
        }
        _ => fail!()
    }
    match x {
        [..a, b, c] => {
            assert_eq!(a, &[1]);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
        _ => fail!()
    }
    match x {
        [a, ..b, c] => {
            assert_eq!(a, 1);
            assert_eq!(b, &[2]);
            assert_eq!(c, 3);
        }
        _ => fail!()
    }
    match x {
        [a, b, c] => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
        _ => fail!()
    }
}

fn c() {
    let x = [1];
    match x {
        [2, .. _] => fail!(),
        [.. _] => ()
    }
}

fn d() {
    let x = [1, 2, 3];
    let branch = match x {
        [1, 1, .. _] => 0,
        [1, 2, 3, .. _] => 1,
        [1, 2, .. _] => 2,
        _ => 3
    };
    assert_eq!(branch, 1);
}

pub fn main() {
    a();
    b();
    c();
    d();
}
