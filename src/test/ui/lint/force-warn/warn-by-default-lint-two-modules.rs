// --force-warn $LINT causes $LINT (which is warn-by-default) to warn
// despite being allowed in one submodule (but not the other)
// compile-flags: --force-warn dead_code
// check-pass

mod one {
    #![allow(dead_code)]

    fn dead_function() {}
    //~^ WARN function is never used
}

mod two {
    fn dead_function() {}
    //~^ WARN function is never used
}

fn main() {}
