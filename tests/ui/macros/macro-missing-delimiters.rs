macro_rules! baz(
    baz => () //~ ERROR invalid macro matcher;
);

fn main() {
    baz!(baz);
}
