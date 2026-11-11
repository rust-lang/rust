// Check that when making a ref mut binding with type `&mut T`, the
// type `T` must match precisely the type `U` of the value being
// matched, and in particular cannot be some supertype of `U`. Issue
// #23116. This test focuses on a `let`.

#![allow(dead_code)]
struct S<'b>(&'b i32);
impl<'b> S<'b> {
    fn bar<'a>(&'a mut self) -> &'a mut &'a i32 {
        let ref mut x = self.0;
        x
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {}
