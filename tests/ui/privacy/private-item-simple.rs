mod a {
    fn f() {}
}

fn main() {
    a::f(); //~ ERROR function `f` is private
}
