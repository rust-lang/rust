macro_rules! foo {
    ($a:ident) => ();
    ($a:ident, $b:ident) => ();
    ($a:ident, $b:ident, $c:ident) => ();
    ($a:ident, $b:ident, $c:ident, $d:ident) => ();
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => ();
}

macro_rules! bar {
    ($lvl:expr, $($arg:tt)+) => {}
}

macro_rules! check {
    ($ty:ty, $expected:expr) => {};
    ($ty_of:expr, $expected:expr) => {};
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
    bar!(Level::Error, );
    //~^ ERROR unexpected end of macro invocation
    check!(<str as Debug>::fmt, "fmt");
    check!(<str as Debug>::fmt, "fmt",);
    //~^ ERROR no rules expected the token `,`
}
