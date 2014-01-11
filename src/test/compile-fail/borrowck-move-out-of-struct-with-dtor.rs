struct S {f:~str}
impl Drop for S {
    fn drop(&mut self) { println!("{}", self.f); }
}

fn move_in_match() {
    match S {f:~"foo"} {
        S {f:_s} => {}
        //~^ ERROR cannot move out of type `S`, which defines the `Drop` trait
    }
}

fn move_in_let() {
    let S {f:_s} = S {f:~"foo"};
    //~^ ERROR cannot move out of type `S`, which defines the `Drop` trait
}

fn move_in_fn_arg(S {f:_s}: S) {
    //~^ ERROR cannot move out of type `S`, which defines the `Drop` trait
}

fn main() {}
