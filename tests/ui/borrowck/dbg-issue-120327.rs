//! Diagnostic test for <https://github.com/rust-lang/rust/issues/120327>: suggest borrowing
//! variables passed to `dbg!` that are later used.
//@ dont-require-annotations: HELP

fn s() -> String {
    let a = String::new();
    dbg!(a); //~ HELP consider borrowing instead of transferring ownership
    return a; //~ ERROR use of moved value:
}

fn m() -> String {
    let a = String::new();
    dbg!(1, 2, a, 1, 2); //~ HELP consider borrowing instead of transferring ownership
    return a; //~ ERROR use of moved value:
}

fn t(a: String) -> String {
    let b: String = "".to_string();
    dbg!(a, b); //~ HELP consider borrowing instead of transferring ownership
    return b; //~ ERROR use of moved value:
}

fn x(a: String) -> String {
    let b: String = "".to_string();
    dbg!(a, b); //~ HELP consider borrowing instead of transferring ownership
    return a; //~ ERROR use of moved value:
}

fn two_of_them(a: String) -> String {
    dbg!(a, a); //~ ERROR use of moved value
    //~| HELP consider borrowing instead of transferring ownership
    //~| HELP consider borrowing instead of transferring ownership
    return a; //~ ERROR use of moved value
}

fn get_expr(_s: String) {}

// The suggestion is purely syntactic; applying it here will result in a type error.
fn test() {
    let a: String = "".to_string();
    let _res = get_expr(dbg!(a)); //~ HELP consider borrowing instead of transferring ownership
    let _l = a.len(); //~ ERROR borrow of moved value
}

fn main() {}
