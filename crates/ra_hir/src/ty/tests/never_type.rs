use super::type_at;

#[test]
fn infer_never1() {
    let t = type_at(
        r#"
//- /main.rs
fn test() {
    let t = return;
    t<|>;
}
"#,
    );
    assert_eq!(t, "!");
}

#[test]
fn infer_never2() {
    let t = type_at(
        r#"
//- /main.rs
fn gen<T>() -> T { loop {} }

fn test() {
    let a = gen();
    if false { a } else { loop {} };
    a<|>;
}
"#,
    );
    assert_eq!(t, "!");
}

#[test]
fn infer_never3() {
    let t = type_at(
        r#"
//- /main.rs
fn gen<T>() -> T { loop {} }

fn test() {
    let a = gen();
    if false { loop {} } else { a };
    a<|>;
}
"#,
    );
    assert_eq!(t, "!");
}

#[test]
fn never_type_in_generic_args() {
    let t = type_at(
        r#"
//- /main.rs
enum Option<T> { None, Some(T) }

fn test() {
    let a = if true { Option::None } else { Option::Some(return) };
    a<|>;
}
"#,
    );
    assert_eq!(t, "Option<!>");
}

#[test]
fn never_type_can_be_reinferred1() {
    let t = type_at(
        r#"
//- /main.rs
fn gen<T>() -> T { loop {} }

fn test() {
    let a = gen();
    if false { loop {} } else { a };
    a<|>;
    if false { a };
}
"#,
    );
    assert_eq!(t, "()");
}

#[test]
fn never_type_can_be_reinferred2() {
    let t = type_at(
        r#"
//- /main.rs
enum Option<T> { None, Some(T) }

fn test() {
    let a = if true { Option::None } else { Option::Some(return) };
    a<|>;
    match 42 {
        42 => a,
        _ => Option::Some(42),
    };
}
"#,
    );
    assert_eq!(t, "Option<i32>");
}
#[test]
fn never_type_can_be_reinferred3() {
    let t = type_at(
        r#"
//- /main.rs
enum Option<T> { None, Some(T) }

fn test() {
    let a = if true { Option::None } else { Option::Some(return) };
    a<|>;
    match 42 {
        42 => a,
        _ => Option::Some("str"),
    };
}
"#,
    );
    assert_eq!(t, "Option<&str>");
}

#[test]
fn match_no_arm() {
    let t = type_at(
        r#"
//- /main.rs
enum Void {}

fn test(a: Void) {
    let t = match a {};
    t<|>;
}
"#,
    );
    assert_eq!(t, "!");
}

#[test]
fn if_never() {
    let t = type_at(
        r#"
//- /main.rs
fn test() {
    let i = if true {
        loop {}
    } else {
        3.0
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "f64");
}

#[test]
fn if_else_never() {
    let t = type_at(
        r#"
//- /main.rs
fn test(input: bool) {
    let i = if input {
        2.0
    } else {
        return
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "f64");
}

#[test]
fn match_first_arm_never() {
    let t = type_at(
        r#"
//- /main.rs
fn test(a: i32) {
    let i = match a {
        1 => return,
        2 => 2.0,
        3 => loop {},
        _ => 3.0,
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "f64");
}

#[test]
fn match_second_arm_never() {
    let t = type_at(
        r#"
//- /main.rs
fn test(a: i32) {
    let i = match a {
        1 => 3.0,
        2 => loop {},
        3 => 3.0,
        _ => return,
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "f64");
}

#[test]
fn match_all_arms_never() {
    let t = type_at(
        r#"
//- /main.rs
fn test(a: i32) {
    let i = match a {
        2 => return,
        _ => loop {},
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "!");
}

#[test]
fn match_no_never_arms() {
    let t = type_at(
        r#"
//- /main.rs
fn test(a: i32) {
    let i = match a {
        2 => 2.0,
        _ => 3.0,
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "f64");
}
