struct S(~str);
impl Drop for S {
    fn drop(&mut self) { }
}

fn move_in_match() {
    match S(~"foo") {
        S(_s) => {}
        //~^ ERROR cannot move out of type `S`, which defines the `Drop` trait
    }
}

fn move_in_let() {
    let S(_s) = S(~"foo");
    //~^ ERROR cannot move out of type `S`, which defines the `Drop` trait
}

fn move_in_fn_arg(S(_s): S) {
    //~^ ERROR cannot move out of type `S`, which defines the `Drop` trait
}

fn main() {}
