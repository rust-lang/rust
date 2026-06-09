pub trait Foo {}

struct Bar;
struct Baz;

impl Foo for Bar { }
impl Foo for Baz { }

fn not_all_paths(a: &str) -> u32 { //~ ERROR mismatched types
    match a {
        "baz" => 0,
        _ => 1,
    };
}

fn right(b: &str) -> Box<dyn Foo> {
    match b {
        "baz" => Box::new(Baz),
        _ => Box::new(Bar),
    }
}

fn wrong(c: &str) -> Box<dyn Foo> { //~ ERROR mismatched types
    match c {
        "baz" => Box::new(Baz),
        _ => Box::new(Bar), //~ ERROR `match` arms have incompatible types
    };
}

fn main() {}
