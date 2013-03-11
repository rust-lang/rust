fn a() {
    let x = [1];
    match x {
        [_, _, _, _, _, .._] => ::core::util::unreachable(),
        [.._, _, _, _, _] => ::core::util::unreachable(),
        [_, .._, _, _] => ::core::util::unreachable(),
        [_, _] => ::core::util::unreachable(),
        [a] => {
            fail_unless!(a == 1);
        }
        [] => ::core::util::unreachable()
    }
}

fn b() {
    let x = [1, 2, 3];
    match x {
        [a, b, ..c] => {
            fail_unless!(a == 1);
            fail_unless!(b == 2);
            fail_unless!(c == &[3]);
        }
        _ => fail!()
    }
    match x {
        [..a, b, c] => {
            fail_unless!(a == &[1]);
            fail_unless!(b == 2);
            fail_unless!(c == 3);
        }
        _ => fail!()
    }
    match x {
        [a, ..b, c] => {
            fail_unless!(a == 1);
            fail_unless!(b == &[2]);
            fail_unless!(c == 3);
        }
        _ => fail!()
    }
    match x {
        [a, b, c] => {
            fail_unless!(a == 1);
            fail_unless!(b == 2);
            fail_unless!(c == 3);
        }
        _ => fail!()
    }
}

pub fn main() {
    a();
    b();
}
