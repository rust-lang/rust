fn a() {
    let x = [1, 2, 3];
    match x {
        [1, 2, 4] => unreachable!(),
        [0, 2, 3, .._] => unreachable!(),
        [0, .._, 3] => unreachable!(),
        [0, .._] => unreachable!(),
        [1, 2, 3] => (),
        [_, _, _] => unreachable!(),
    }
    match x {
        [.._] => (),
    }
    match x {
        [_, _, _, .._] => (),
    }
    match x {
        [a, b, c] => {
            assert_eq!(1, a);
            assert_eq!(2, b);
            assert_eq!(3, c);
        }
    }
}

pub fn main() {
    a();
}
