macro_rules! foo(
    ($x:foo) => ()
    //~^ ERROR invalid fragment specifier
);

fn main() {
    foo!(foo);
}
