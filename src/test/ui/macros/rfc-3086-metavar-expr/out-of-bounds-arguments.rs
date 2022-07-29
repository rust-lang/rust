#![feature(macro_metavar_expr)]

macro_rules! a {
    ( $( { $( [ $( ( $( $foo:ident )* ) )* ] )* } )* ) => {
        (
            ${count(foo, 0)},
            ${count(foo, 10)},
            //~^ ERROR depth parameter on meta-variable expression `count` must be less than 4
        )
    };
}

macro_rules! b {
    ( $( { $( [ $( $foo:ident )* ] )* } )* ) => {
        (
            $( $( $(
                ${ignore(foo)}
                ${index(0)},
                ${index(10)},
                //~^ ERROR depth parameter on meta-variable expression `index` must be less than 3
            )* )* )*
        )
    };
}

macro_rules! c {
    ( $( { $( $foo:ident )* } )* ) => {
        (
            $( $(
                ${ignore(foo)}
                ${length(0)}
                ${length(10)}
                //~^ ERROR depth parameter on meta-variable expression `length` must be less than 2
            )* )*
        )
    };
}

fn main() {
    a!( { [ (a) ] [ (b c) ] } );
    b!( { [ a b ] } );
    c!({ a });
}
