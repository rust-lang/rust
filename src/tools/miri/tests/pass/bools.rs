fn boolean() -> bool {
    true
}

fn if_false() -> i64 {
    let c = false;
    if c { 1 } else { 0 }
}

fn if_true() -> i64 {
    let c = true;
    if c { 1 } else { 0 }
}

fn match_bool() -> i16 {
    let b = true;
    match b {
        true => 1,
        _ => 0,
    }
}

fn main() {
    assert!(boolean());
    assert_eq!(if_false(), 0);
    assert_eq!(if_true(), 1);
    assert_eq!(match_bool(), 1);
    assert_eq!(true == true, true);
}
