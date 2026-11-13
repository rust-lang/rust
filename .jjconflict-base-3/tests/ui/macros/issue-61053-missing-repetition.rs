#![deny(meta_variable_misuse)]

macro_rules! foo {
    () => {};
    ($( $i:ident = $($j:ident),+ );*) => { $( $i = $j; )* };
    //~^ ERROR variable `j` is still repeating
}

macro_rules! bar {
    () => {};
    (test) => {
        macro_rules! nested {
            () => {};
            ($( $i:ident = $($j:ident),+ );*) => { $( $i = $j; )* };
            //~^ ERROR variable `j` is still repeating
        }
    };
    ( $( $i:ident = $($j:ident),+ );* ) => {
        $(macro_rules! $i {
            () => { $j }; //~ ERROR variable `j` is still repeating
        })*
    };
}

fn main() {
    foo!();
    bar!();
}
