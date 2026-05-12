trait Bar<'a> {
    const STATIC: &'a str;
}

struct A;
impl Bar<'_> for A {
    const STATIC: &str = "";
    //~^ ERROR missing lifetime specifier
}

struct B;
impl Bar<'static> for B {
    const STATIC: &str = "";
}

struct C;
impl Bar<'_> for C {
    // make  ^^ not cause
    const STATIC: &'static str = {
        struct B;
        impl Bar<'static> for B {
            const STATIC: &str = "";
            //            ^ to emit a future incompat warning
        }
        ""
    };
}

fn main() {}
