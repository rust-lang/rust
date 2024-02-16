//@ run-pass
//
// from issue #93951, where borrowck complained the temporary that `foo(&x)` was stored in was to
// be dropped sometime after `x` was. It then suggested adding a semicolon that was already there.


use std::fmt::Debug;

fn foo<'a>(x: &'a str) -> Result<impl Debug + 'a, ()> {
    Ok(x)
}

fn let_else() {
    let x = String::from("Hey");
    let Ok(_) = foo(&x) else { return };
}

fn if_let() {
    let x = String::from("Hey");
    let _ = if let Ok(s) = foo(&x) { s } else { return };
}

fn main() {
    let_else();
    if_let();
}
