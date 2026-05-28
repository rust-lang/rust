//@ check-pass

pub struct S<'a, 'e>(&'a str, &'e str);

pub type T<'a, 'e> = std::collections::HashMap<S<'a, 'e>, ()>;

impl<'e, 'a: 'e> S<'a, 'e> {
    pub fn foo(_a: &str, _b: &str, _map: &T) {}
}

fn main() {}
