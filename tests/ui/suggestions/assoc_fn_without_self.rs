fn main() {}

struct S;

impl S {
    fn foo() {}

    fn bar(&self) {}

    fn baz(a: u8, b: u8) {}

    fn b() {
        fn c() {
            foo(); //~ ERROR cannot find function `foo`
        }
        foo(); //~ ERROR cannot find function `foo`
        bar(); //~ ERROR cannot find function `bar`
        baz(2, 3); //~ ERROR cannot find function `baz`
    }
    fn d(&self) {
        fn c() {
            foo(); //~ ERROR cannot find function `foo`
        }
        foo(); //~ ERROR cannot find function `foo`
        bar(); //~ ERROR cannot find function `bar`
        baz(2, 3); //~ ERROR cannot find function `baz`
    }
}
