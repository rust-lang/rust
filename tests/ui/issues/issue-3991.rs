//@ check-pass
#![allow(dead_code)]


struct HasNested {
    nest: Vec<Vec<isize> > ,
}

impl HasNested {
    fn method_push_local(&mut self) {
        self.nest[0].push(0);
    }
}

pub fn main() {}
