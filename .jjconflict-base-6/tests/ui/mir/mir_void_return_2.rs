//@ run-pass
fn nil() {}

fn mir(){
    nil()
}

pub fn main() {
    mir();
}
