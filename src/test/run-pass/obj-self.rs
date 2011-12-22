

fn main() {
    obj foo() {
        fn m1() { #debug("hi!"); }
        fn m2() { self.m1(); }
    }
    let a = foo();
    a.m1();
    a.m2();
}
