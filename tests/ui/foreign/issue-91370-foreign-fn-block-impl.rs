// Regression test for issue #91370.

extern "C" {
    //~^ NOTE_NONVIRAL `extern` blocks define existing foreign functions
    fn f() {
        //~^ ERROR incorrect function inside `extern` block
        //~| NOTE_NONVIRAL cannot have a body
        impl Copy for u8 {}
    }
}

fn main() {}
