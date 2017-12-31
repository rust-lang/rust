// rustfmt-spaces_within_parens_and_brackets: true
// Spaces within parens and brackets

fn lorem< T: Eq >( t: T ) {
    let lorem = ( ipsum, dolor );
    let lorem: [ usize; 2 ] = [ ipsum, dolor ];
}

enum E {
    A( u32 ),
    B( u32, u32 ),
    C( u32, u32, u32 ),
    D(),
}

struct TupleStruct0();
struct TupleStruct1( u32 );
struct TupleStruct2( u32, u32 );

fn fooEmpty() {}

fn foo( e: E, _: u32 ) -> ( u32, u32 ) {
    // Tuples
    let t1 = ();
    let t2 = ( 1, );
    let t3 = ( 1, 2 );

    let ts0 = TupleStruct0();
    let ts1 = TupleStruct1( 1 );
    let ts2 = TupleStruct2( 1, 2 );

    // Tuple pattern
    let ( a, b, c ) = ( 1, 2, 3 );

    // Expressions
    let x = ( 1 + 2 ) * ( 3 );

    // Function call
    fooEmpty();
    foo( 1, 2 );

    // Pattern matching
    match e {
        A( _ ) => (),
        B( _, _ ) => (),
        C( .. ) => (),
        D => (),
    }

    ( 1, 2 )
}

struct Foo< T > {
    i: T,
}

struct Bar< T, E > {
    i: T,
    e: E,
}

struct Foo< 'a > {
    i: &'a str,
}

enum E< T > {
    T( T ),
}

enum E< T, S > {
    T( T ),
    S( S ),
}

fn foo< T >( a: T ) {
    foo::< u32 >( 10 );
}

fn foo< T, E >( a: T, b: E ) {
    foo::< u32, str >( 10, "bar" );
}

fn foo< T: Send, E: Send >( a: T, b: E ) {
    foo::< u32, str >( 10, "bar" );

    let opt: Option< u32 >;
    let res: Result< u32, String >;
}

fn foo< 'a >( a: &'a str ) {
    foo( "foo" );
}

fn foo< 'a, 'b >( a: &'a str, b: &'b str ) {
    foo( "foo", "bar" );
}

impl Foo {
    fn bar() {
        < Foo as Foo >::bar();
    }
}

trait MyTrait< A, D > {}
impl< A: Send, D: Send > MyTrait< A, D > for Foo {}

fn foo()
where
    for< 'a > u32: 'a,
{
}

fn main() {
    let arr: [ i32; 5 ] = [ 1, 2, 3, 4, 5 ];
    let arr: [ i32; 500 ] = [ 0; 500 ];

    let v = vec![ 1, 2, 3 ];
    assert_eq!( arr, [ 1, 2, 3 ] );

    let i = arr[ 0 ];

    let slice = &arr[ 1..2 ];

    let line100_________________________________________________________________________ = [ 1, 2 ];
    let line101__________________________________________________________________________ =
        [ 1, 2 ];
    let line102___________________________________________________________________________ =
        [ 1, 2 ];
    let line103____________________________________________________________________________ =
        [ 1, 2 ];
    let line104_____________________________________________________________________________ =
        [ 1, 2 ];

    let line100_____________________________________________________________________ = vec![ 1, 2 ];
    let line101______________________________________________________________________ =
        vec![ 1, 2 ];
    let line102_______________________________________________________________________ =
        vec![ 1, 2 ];
    let line103________________________________________________________________________ =
        vec![ 1, 2 ];
    let line104_________________________________________________________________________ =
        vec![ 1, 2 ];
}

fn f( slice: &[ i32 ] ) {}
