fn with(f: &fn(&~str)) {}

fn arg_item(&_x: &~str) {}
    //~^ ERROR cannot move out of dereference of & pointer

fn arg_closure() {
    with(|&_x| ())
    //~^ ERROR cannot move out of dereference of & pointer
}

fn let_pat() {
    let &_x = &~"hi";
    //~^ ERROR cannot move out of dereference of & pointer
}

pub fn main() {}
