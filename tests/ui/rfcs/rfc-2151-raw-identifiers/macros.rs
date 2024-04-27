//@ run-pass
#![feature(decl_macro)]

macro_rules! r#struct {
    ($r#struct:expr) => { $r#struct }
}

macro_rules! old_macro {
    ($a:expr) => {$a}
}

macro r#decl_macro($r#fn:expr) {
    $r#fn
}

macro passthrough($id:ident) {
    $id
}

macro_rules! test_pat_match {
    (a) => { 6 };
    (r#a) => { 7 };
}

pub fn main() {
    r#println!("{struct}", r#struct = 1);
    assert_eq!(2, r#struct!(2));
    assert_eq!(3, r#old_macro!(3));
    assert_eq!(4, decl_macro!(4));

    let r#match = 5;
    assert_eq!(5, passthrough!(r#match));

    assert_eq!("r#struct", stringify!(r#struct));

    assert_eq!(6, test_pat_match!(a));
    assert_eq!(7, test_pat_match!(r#a));
}
