struct Foo (
    fn([u8; |x: u8| {}]), //~ ERROR mismatched types
);

fn main() {}
