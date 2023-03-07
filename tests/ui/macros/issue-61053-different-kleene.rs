#![deny(meta_variable_misuse)]

macro_rules! foo {
    () => {};
    ( $( $i:ident = $($j:ident),+ );* ) => { $( $( $i = $j; )* )* };
    //~^ ERROR meta-variable repeats with
    ( $( $($j:ident),+ );* ) => { $( $( $j; )+ )+ }; //~ERROR meta-variable repeats with
}

macro_rules! bar {
    () => {};
    (test) => {
        macro_rules! nested {
            () => {};
            ( $( $i:ident = $($j:ident),+ );* ) => { $( $( $i = $j; )* )* };
            //~^ ERROR meta-variable repeats with
            ( $( $($j:ident),+ );* ) => { $( $( $j; )+ )+ }; //~ERROR meta-variable repeats with
        }
    };
    ( $( $i:ident = $($j:ident),+ );* ) => {
        $(macro_rules! $i {
            () => { 0 $( + $j )* }; //~ ERROR meta-variable repeats with
        })*
    };
}

fn main() {
    foo!();
    bar!();
}
