#![feature(macro_metavar_expr)]

macro_rules! a {
    ( $( { $( [ $( ( $( $foo:ident )* ) )* ] )* } )* ) => {
        (
            ${count($foo, 0)},
            ${count($foo, 10)},
            //~^ ERROR depth parameter of meta-variable expression `count` must be less than 4
        )
    };
}

macro_rules! b {
    ( $( { $( [ $( $foo:ident )* ] )* } )* ) => {
        (
            $( $( $(
                ${ignore($foo)}
                ${index(0)},
                ${index(10)},
                //~^ ERROR depth parameter of meta-variable expression `index` must be less than 3
            )* )* )*
        )
    };
}

macro_rules! c {
    ( $( { $( $foo:ident )* } )* ) => {
        (
            $( $(
                ${ignore($foo)}
                ${len(0)}
                ${len(10)}
                //~^ ERROR depth parameter of meta-variable expression `len` must be less than 2
            )* )*
        )
    };
}

fn main() {
    a!( { [ (a) ] [ (b c) ] } );
    b!( { [ a b ] } );
    c!({ a });
}
