fn f() -> ! {
    ret 42; //! ERROR expected `_|_` but found `int` (types differ)
    fail; //! WARNING unreachable statement
}
fn main() { }
