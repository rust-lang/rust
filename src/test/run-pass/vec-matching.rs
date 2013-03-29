fn a() {
    let x = [1];
    match x {
        [_, _, _, _, _, .._] => ::core::util::unreachable(),
        [.._, _, _, _, _] => ::core::util::unreachable(),
        [_, .._, _, _] => ::core::util::unreachable(),
        [_, _] => ::core::util::unreachable(),
        [a] => {
            assert!(a == 1);
        }
        [] => ::core::util::unreachable()
    }
}

fn b() {
    let x = [1, 2, 3];
    match x {
        [a, b, ..c] => {
            assert!(a == 1);
            assert!(b == 2);
            assert!(c == &[3]);
        }
        _ => fail!()
    }
    match x {
        [..a, b, c] => {
            assert!(a == &[1]);
            assert!(b == 2);
            assert!(c == 3);
        }
        _ => fail!()
    }
    match x {
        [a, ..b, c] => {
            assert!(a == 1);
            assert!(b == &[2]);
            assert!(c == 3);
        }
        _ => fail!()
    }
    match x {
        [a, b, c] => {
            assert!(a == 1);
            assert!(b == 2);
            assert!(c == 3);
        }
        _ => fail!()
    }
}

pub fn main() {
    a();
    b();
}
