#![deny(meta_variable_misuse)]

macro_rules! foo {
    () => {};
    ($( $i:ident = $($j:ident),+ );*) => { $( $( $i = $k; )+ )* };
    //~^ ERROR unknown macro variable
}

macro_rules! bar {
    () => {};
    (test) => {
        macro_rules! nested {
            () => {};
            ($( $i:ident = $($j:ident),+ );*) => { $( $( $i = $k; )+ )* };
            //~^ ERROR unknown macro variable
        }
    };
    ( $( $i:ident = $($j:ident),+ );* ) => {
        $(macro_rules! $i {
            () => { $( $i = $k)+ }; //~ ERROR unknown macro variable
        })*
    };
}

fn main() {
    foo!();
    bar!();
}
