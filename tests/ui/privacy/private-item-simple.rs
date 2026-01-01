//! regression test for issue #3993

mod a {
    fn f() {}
}

fn main() {
    a::f(); //~ ERROR function `f` is private
}

fn foo() {
    use a::f; //~ ERROR function `f` is private
}
