//@ run-pass
fn tuple() {
    let x = (1, 2, 3);
    match x {
        (1, 2, 4) => unreachable!(),
        (0, 2, 3, ..) => unreachable!(),
        (0, .., 3) => unreachable!(),
        (0, ..) => unreachable!(),
        (1, 2, 3) => (),
        (_, _, _) => unreachable!(),
    }
    match x {
        (..) => (),
    }
    match x {
        (_, _, _, ..) => (),
    }
    match x {
        (a, b, c) => {
            assert_eq!(1, a);
            assert_eq!(2, b);
            assert_eq!(3, c);
        }
    }
}

fn tuple_struct() {
    struct S(u8, u8, u8);

    let x = S(1, 2, 3);
    match x {
        S(1, 2, 4) => unreachable!(),
        S(0, 2, 3, ..) => unreachable!(),
        S(0, .., 3) => unreachable!(),
        S(0, ..) => unreachable!(),
        S(1, 2, 3) => (),
        S(_, _, _) => unreachable!(),
    }
    match x {
        S(..) => (),
    }
    match x {
        S(_, _, _, ..) => (),
    }
    match x {
        S(a, b, c) => {
            assert_eq!(1, a);
            assert_eq!(2, b);
            assert_eq!(3, c);
        }
    }
}

fn main() {
    tuple();
    tuple_struct();
}
