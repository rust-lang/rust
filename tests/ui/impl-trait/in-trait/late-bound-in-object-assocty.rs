// Test for issue #132429
//@compile-flags: -Zunstable-options --edition=2024

trait ThreeCellFragment {
    fn ext_cells<'a>(
        &'a self,
    ) -> dyn core::future::Future<Output = impl IntoIterator<Item = u32>> + 'a {
        //~^ ERROR mismatched types
        //~| ERROR return type cannot have an unboxed trait object
    }
}

fn main() {}
