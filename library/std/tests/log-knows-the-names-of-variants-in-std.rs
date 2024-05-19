#![allow(non_camel_case_types)]
#![allow(dead_code)]

#[derive(Clone, Debug)]
enum foo {
    a(usize),
    b(String),
}

fn check_log<T: std::fmt::Debug>(exp: String, v: T) {
    assert_eq!(exp, format!("{:?}", v));
}

#[test]
fn log_knows_the_names_of_variants_in_std() {
    let mut x = Some(foo::a(22));
    let exp = "Some(a(22))".to_string();
    let act = format!("{:?}", x);
    assert_eq!(act, exp);
    check_log(exp, x);

    x = None;
    let exp = "None".to_string();
    let act = format!("{:?}", x);
    assert_eq!(act, exp);
    check_log(exp, x);
}
