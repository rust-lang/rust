// error-pattern: mismatched types
use foo::foo;

use std::borrow::Cow;

fn use_string(_x: String) {}

fn make_cow() -> Cow<'static, str> {
    Cow::Borrowed("foo")
}

fn main() {
    use_string(make_cow());
}