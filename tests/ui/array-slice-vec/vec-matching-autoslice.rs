//@ run-pass

pub fn main() {
    let x = [1, 2, 3];
    match x {
        [2, _, _] => panic!(),
        [1, a, b] => {
            assert_eq!([a, b], [2, 3]);
        }
        [_, _, _] => panic!(),
    }

    let y = ([(1, true), (2, false)], 0.5f64);
    match y {
        ([(1, a), (b, false)], _) => {
            assert_eq!(a, true);
            assert_eq!(b, 2);
        }
        ([_, _], 0.5) => panic!(),
        ([_, _], _) => panic!(),
    }
}
