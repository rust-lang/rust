// compile-flags: --crate-type lib -O -C debug-assertions=yes

// Regression test for issue 118786

macro_rules! make_macro {
    ($macro_name:tt) => {
        macro_rules! $macro_name {
            //~^ ERROR: expected identifier, found `(`
            () => {}
        }
    }
}

make_macro!((meow));
