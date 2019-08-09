#[derive(Copy, Clone)]
struct S;

impl S {
    fn mutate(&mut self) {
    }
}

fn func(arg: S) {
    //~^ HELP consider changing this to be mutable
    //~| SUGGESTION mut arg
    arg.mutate();
    //~^ ERROR cannot borrow `arg` as mutable, as it is not declared as mutable
}

fn main() {
    let local = S;
    //~^ HELP consider changing this to be mutable
    //~| SUGGESTION mut local
    local.mutate();
    //~^ ERROR cannot borrow `local` as mutable, as it is not declared as mutable
}
