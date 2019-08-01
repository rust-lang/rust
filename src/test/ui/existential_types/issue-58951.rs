// check-pass
#![feature(existential_type)]

existential type A: Iterator;
fn def_a() -> A { 0..1 }
pub fn use_a() {
    def_a().map(|x| x);
}

fn main() {}
