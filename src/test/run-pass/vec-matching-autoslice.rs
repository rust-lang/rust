pub fn main() {
    let x = @[1, 2, 3];
    match x {
        [2, .._] => fail2!(),
        [1, ..tail] => {
            assert_eq!(tail, [2, 3]);
        }
        [_] => fail2!(),
        [] => fail2!()
    }

    let y = (~[(1, true), (2, false)], 0.5);
    match y {
        ([_, _, _], 0.5) => fail2!(),
        ([(1, a), (b, false), ..tail], _) => {
            assert_eq!(a, true);
            assert_eq!(b, 2);
            assert!(tail.is_empty());
        }
        ([.._tail], _) => fail2!()
    }
}
