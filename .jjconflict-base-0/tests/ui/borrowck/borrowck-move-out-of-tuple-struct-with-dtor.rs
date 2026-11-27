//@ run-rustfix
#![allow(unused)]
struct S(String);
impl Drop for S {
    fn drop(&mut self) { }
}

fn move_in_match() {
    match S("foo".to_string()) {
        //~^ ERROR cannot move out of type `S`, which implements the `Drop` trait
        S(_s) => {}
    }
}

fn move_in_let() {
    let S(_s) = S("foo".to_string());
    //~^ ERROR cannot move out of type `S`, which implements the `Drop` trait
}

fn move_in_fn_arg(S(_s): S) {
    //~^ ERROR cannot move out of type `S`, which implements the `Drop` trait
}

fn main() {}
