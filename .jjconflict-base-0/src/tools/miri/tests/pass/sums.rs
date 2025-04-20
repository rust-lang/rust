#[derive(Debug, PartialEq)]
enum Unit {
    Unit(()), // Force non-C-enum representation.
}

fn return_unit() -> Unit {
    Unit::Unit(())
}

#[derive(Debug, PartialEq)]
enum MyBool {
    False(()), // Force non-C-enum representation.
    True(()),
}

fn return_true() -> MyBool {
    MyBool::True(())
}

fn return_false() -> MyBool {
    MyBool::False(())
}

fn return_none() -> Option<i64> {
    None
}

fn return_some() -> Option<i64> {
    Some(42)
}

fn match_opt_none() -> i8 {
    let x = None;
    match x {
        Some(data) => data,
        None => 42,
    }
}

fn match_opt_some() -> i8 {
    let x = Some(13);
    match x {
        Some(data) => data,
        None => 20,
    }
}

fn two_nones() -> (Option<i16>, Option<i16>) {
    (None, None)
}

fn main() {
    assert_eq!(two_nones(), (None, None));
    assert_eq!(match_opt_some(), 13);
    assert_eq!(match_opt_none(), 42);
    assert_eq!(return_some(), Some(42));
    assert_eq!(return_none(), None);
    assert_eq!(return_false(), MyBool::False(()));
    assert_eq!(return_true(), MyBool::True(()));
    assert_eq!(return_unit(), Unit::Unit(()));
}
