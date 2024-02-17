//@ run-pass
#![allow(non_camel_case_types)]

struct Clam<'a> {
    chowder: &'a isize
}

trait get_chowder<'a> {
    fn get_chowder(&self) -> &'a isize;
}

impl<'a> get_chowder<'a> for Clam<'a> {
    fn get_chowder(&self) -> &'a isize { return self.chowder; }
}

pub fn main() {
    let clam = Clam { chowder: &3 };
    println!("{}", *clam.get_chowder());
    clam.get_chowder();
}
