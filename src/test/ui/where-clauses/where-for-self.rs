// Test that we can quantify lifetimes outside a constraint (i.e., including
// the self type) in a where clause. Specifically, test that we cannot nest
// quantification in constraints (to be clear, there is no reason this should not
// we're testing we don't crash or do something stupid).

trait Bar<'a> {
    fn bar(&self);
}

impl<'a, 'b> Bar<'b> for &'a u32 {
    fn bar(&self) {}
}

fn foo<T>(x: &T)
    where for<'a> &'a T: for<'b> Bar<'b>
    //~^ error: nested quantification of lifetimes
{}

fn main() {}
