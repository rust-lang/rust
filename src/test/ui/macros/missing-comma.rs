macro_rules! foo {
    ($a:ident) => ();
    ($a:ident, $b:ident) => ();
    ($a:ident, $b:ident, $c:ident) => ();
    ($a:ident, $b:ident, $c:ident, $d:ident) => ();
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => ();
}

fn main() {
    println!("{}" a);
    //~^ ERROR expected token: `,`
    foo!(a b);
    //~^ ERROR no rules expected the token `b`
    foo!(a, b, c, d e);
    //~^ ERROR no rules expected the token `e`
    foo!(a, b, c d, e);
    //~^ ERROR no rules expected the token `d`
    foo!(a, b, c d e);
    //~^ ERROR no rules expected the token `d`
}
