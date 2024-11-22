// Test for issue #132429
//@ edition: 2024
//@check-pass

use std::future::Future;

trait Test {
    fn foo<'a>(&'a self) -> Box<dyn Future<Output = impl IntoIterator<Item = u32>>> {
        Box::new(async { [] })
    }
}

fn main() {}
