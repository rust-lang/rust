//@ compile-flags: -Zunpretty=hir
//@ check-pass
//@ edition: 2015

use std::fmt;

pub struct Bar {
    a: String,
    b: u8,
}

impl fmt::Debug for Bar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug_struct_field2_finish(f, "Bar", "a", &self.a, "b", &&self.b)
    }
}

fn debug_struct_field2_finish<'a>(
    name: &str,
    name1: &str,
    value1: &'a dyn fmt::Debug,
    name2: &str,
    value2: &'a dyn fmt::Debug,
) -> fmt::Result
{
    loop {}
}
