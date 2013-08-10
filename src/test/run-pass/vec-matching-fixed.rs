fn a() {
    let x = [1, 2, 3];
    match x {
        [1, 2, 4] => ::std::util::unreachable(),
        [0, 2, 3, .._] => ::std::util::unreachable(),
        [0, .._, 3] => ::std::util::unreachable(),
        [0, .._] => ::std::util::unreachable(),
        [1, 2, 3] => (),
        [_, _, _] => ::std::util::unreachable(),
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
