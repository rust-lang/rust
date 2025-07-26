//! The drop check is currently more permissive when `let` statements have an `else` block, due to
//! scheduling drops for bindings' storage before pattern-matching (#142056).

struct Struct<T>(T);
impl<T> Drop for Struct<T> {
    fn drop(&mut self) {}
}

fn main() {
    {
        // This is an error: `short1` is dead before `long1` is dropped.
        let (mut long1, short1) = (Struct(&0), 1);
        long1.0 = &short1;
        //~^ ERROR `short1` does not live long enough
    }
    {
        // This is OK: `short2`'s storage is live until after `long2`'s drop runs.
        #[expect(irrefutable_let_patterns)]
        let (mut long2, short2) = (Struct(&0), 1) else { unreachable!() };
        long2.0 = &short2;
    }
    {
        // Sanity check: `short3`'s drop is significant; it's dropped before `long3`:
        let tmp = Box::new(0);
        #[expect(irrefutable_let_patterns)]
        let (mut long3, short3) = (Struct(&tmp), Box::new(1)) else { unreachable!() };
        long3.0 = &short3;
        //~^ ERROR `short3` does not live long enough
    }
}
