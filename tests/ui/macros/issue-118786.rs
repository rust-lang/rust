//@ compile-flags: --crate-type lib -O -C debug-assertions=yes
//@ dont-require-annotations: NOTE

// Regression test for issue 118786

macro_rules! make_macro {
    ($macro_name:tt) => {
        macro_rules! $macro_name {
        //~^ ERROR macro expansion ignores `{` and any tokens following
        //~| ERROR cannot find macro `macro_rules` in this scope
        //~| NOTE put a macro name here
            () => {}
        }
    }
}

make_macro!((meow));
//~^ ERROR macros that expand to items must be delimited with braces or followed by a semicolon
