// xfail-pretty - token trees can't pretty print

macro_rules! myfn(
    ( $f:ident, ( $( $x:ident ),* ), $body:block ) => (
        fn $f( $( $x : int),* ) -> int $body
    )
)

myfn!(add, (a,b), { return a+b; } )

fn main() {

    macro_rules! mylet(
        ($x:ident, $val:expr) => (
            let $x = $val;
        )
    );

    mylet!(y, 8*2);
    assert(y == 16);

    myfn!(mult, (a,b), { a*b } );

    assert (mult(2, add(4,4)) == 16);

    macro_rules! actually_an_expr_macro (
        () => ( 16 )
    )

    assert { actually_an_expr_macro!() } == 16;

}
