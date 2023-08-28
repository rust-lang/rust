#![feature(macro_metavar_expr)]

macro_rules! foo {
    ($($t:ident)*) => { ${count(t, 4294967296)} };
    //~^ ERROR related fragment that refers the `count` meta-variable expression was not found
}

macro_rules! bar {
    ( $( { $( [ $( ( $( $t:ident )* ) )* ] )* } )* ) => { ${count(t, 4294967296)} }
    //~^ ERROR related fragment that refers the `count` meta-variable expression was not found
}

fn test() {
    foo!();
    bar!( { [] [] } );
}

fn main() {
}
