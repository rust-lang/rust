//@ revisions: rpass1 rpass2

#[cfg(rpass1)]
pub trait Something {
    fn foo();
}

#[cfg(rpass2)]
pub struct Something {
    pub foo: u8,
}

fn main() {}
