//@ run-pass
macro_rules! myfn {
    ( $f:ident, ( $( $x:ident ),* ), $body:block ) => (
        fn $f( $( $x : isize),* ) -> isize $body
    )
}

myfn!(add, (a,b), { return a+b; } );

pub fn main() {

    macro_rules! mylet {
        ($x:ident, $val:expr) => (
            let $x = $val;
        )
    }

    mylet!(y, 8*2);
    assert_eq!(y, 16);

    myfn!(mult, (a,b), { a*b } );

    assert_eq!(mult(2, add(4,4)), 16);

    macro_rules! actually_an_expr_macro {
        () => ( 16 )
    }

    assert_eq!({ actually_an_expr_macro!() }, 16);

}
