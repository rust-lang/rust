// Regression test for issue #91370.

extern "C" {
    //~^ NOTE `extern` blocks define existing foreign functions
    fn f() {
        //~^ ERROR incorrect function inside `extern` block
        //~| NOTE cannot have a body
        //~| NOTE for more information, visit https://doc.rust-lang.org/std/keyword.extern.html
        impl Copy for u8 {}
    }
}

fn main() {}
