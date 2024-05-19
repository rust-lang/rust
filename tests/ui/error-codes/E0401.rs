trait Baz<T> {}

fn foo<T>(x: T) {
    fn bfnr<U, V: Baz<U>, W: Fn()>(y: T) { //~ ERROR E0401
    }
    fn baz<U,
           V: Baz<U>,
           W: Fn()>
           (y: T) { //~ ERROR E0401
    }
    bfnr(x);
}


struct A<T> {
    inner: T,
}

impl<T> Iterator for A<T> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        fn helper(sel: &Self) -> u8 { //~ ERROR E0401
            unimplemented!();
        }
        Some(helper(self))
    }
}

fn main() {
}
