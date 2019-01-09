// run-pass
macro_rules! var {
    ( $name:ident ) => ( $name );
}

pub fn main() {
    let x = [ 3, 3, 3 ];
    for var!(i) in &x {
        assert_eq!(*i, 3);
    }
}
