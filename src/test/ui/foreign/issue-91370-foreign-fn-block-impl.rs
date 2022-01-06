// Regression test for issue #91370.

extern {
    //~^ `extern` blocks define existing foreign functions
    fn f() {
        //~^ incorrect function inside `extern` block
        //~| cannot have a body
        impl Copy for u8 {}
    }
}

fn main() {}
