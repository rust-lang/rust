fn main() {
    concat!(b'f');  //~ ERROR: cannot concatenate a byte string literal
    concat!(b"foo");  //~ ERROR: cannot concatenate a byte string literal
    concat!(foo);   //~ ERROR: expected a literal
    concat!(foo()); //~ ERROR: expected a literal
}
