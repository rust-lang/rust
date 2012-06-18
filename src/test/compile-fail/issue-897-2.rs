fn g() -> ! { fail; }
fn f() -> ! {
    ret 42i; //! ERROR expected `_|_` but found `int`
    g(); //! WARNING unreachable statement
}
fn main() { }
