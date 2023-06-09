// run-pass
// #30527 - We were not generating arms with guards in certain cases.

fn match_with_guard(x: Option<i8>) -> i8 {
    match x {
        Some(xyz) if xyz > 100 => 0,
        Some(_) => -1,
        None => -2
    }
}

fn main() {
    assert_eq!(match_with_guard(Some(111)), 0);
    assert_eq!(match_with_guard(Some(2)), -1);
    assert_eq!(match_with_guard(None), -2);
}
