//@ check-pass

pub struct A<'a>(&'a ());

impl<'a> A<'a> {
    const N: usize = 68;

    pub fn foo(&self) {
        let _b = [0; Self::N];
    }
}

fn main() {}
