#[derive(Copy, Clone)]
struct S;

impl S {
    fn mutate(&mut self) {
    }
}

fn func(arg: S) {
    //~^ consider changing this to `mut arg`
    arg.mutate();
    //~^ ERROR cannot borrow immutable argument
    //~| cannot borrow mutably
}

fn main() {
    let local = S;
    //~^ consider changing this to `mut local`
    local.mutate();
    //~^ ERROR cannot borrow immutable local variable
    //~| cannot borrow mutably
}
