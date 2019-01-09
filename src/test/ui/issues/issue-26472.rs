mod sub {
    pub struct S { len: usize }
    impl S {
        pub fn new() -> S { S { len: 0 } }
        pub fn len(&self) -> usize { self.len }
    }
}

fn main() {
    let s = sub::S::new();
    let v = s.len; //~ ERROR field `len` of struct `sub::S` is private
    s.len = v; //~ ERROR field `len` of struct `sub::S` is private
}
