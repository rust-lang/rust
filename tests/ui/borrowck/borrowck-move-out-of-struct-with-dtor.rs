//@ run-rustfix
#![allow(unused)]
struct S {f:String}
impl Drop for S {
    fn drop(&mut self) { println!("{}", self.f); }
}

fn move_in_match() {
    match (S {f:"foo".to_string()}) {
        //~^ ERROR [E0509]
        S {f:_s} => {}
    }
}

fn move_in_let() {
    let S {f:_s} = S {f:"foo".to_string()};
    //~^ ERROR [E0509]
}

fn move_in_fn_arg(S {f:_s}: S) {
    //~^ ERROR [E0509]
}

fn main() {}
