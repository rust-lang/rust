use std::borrow::BorrowMut;

trait NestedBorrowMut<U, V> {
    fn nested_borrow_mut(&mut self) -> &mut V;
}

impl<T, U, V> NestedBorrowMut<U, V> for T
where
    T: BorrowMut<U>,
    U: BorrowMut<V>, // Error is caused by missing lifetime here
{
    fn nested_borrow_mut(&mut self) -> &mut V {
        let u_ref = self.borrow_mut(); //~ ERROR E0311
        u_ref.borrow_mut() //~ ERROR E0311
    }
}

fn main() {}
