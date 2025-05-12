use std::convert::AsRef;
use std::path::Path;

fn foo11(_bar: &dyn AsRef<Path>, _baz: &str) {}
fn foo12(_bar: &str, _baz: &dyn AsRef<Path>) {}

fn foo21(_bar: &dyn AsRef<str>, _baz: &str) {}
fn foo22(_bar: &str, _baz: &dyn AsRef<str>) {}

fn main() {
    foo11("bar", &"baz"); //~ ERROR the size for values of type
    foo11(&"bar", &"baz");
    foo12(&"bar", "baz"); //~ ERROR the size for values of type
    foo12(&"bar", &"baz");

    foo21("bar", &"baz"); //~ ERROR the size for values of type
    foo21(&"bar", &"baz");
    foo22(&"bar", "baz"); //~ ERROR the size for values of type
    foo22(&"bar", &"baz");
}
