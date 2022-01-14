use std::fmt::Debug;
use std::ptr;
use std::rc::Rc;
use std::sync::Arc;

#[warn(clippy::vtable_address_comparisons)]
#[allow(clippy::borrow_as_ptr)]

fn main() {
    let a: *const dyn Debug = &1 as &dyn Debug;
    let b: *const dyn Debug = &1 as &dyn Debug;

    // These should fail:
    let _ = a == b;
    let _ = a != b;
    let _ = a < b;
    let _ = a <= b;
    let _ = a > b;
    let _ = a >= b;
    ptr::eq(a, b);

    let a = &1 as &dyn Debug;
    let b = &1 as &dyn Debug;
    ptr::eq(a, b);

    let a: Rc<dyn Debug> = Rc::new(1);
    Rc::ptr_eq(&a, &a);

    let a: Arc<dyn Debug> = Arc::new(1);
    Arc::ptr_eq(&a, &a);

    // These should be fine:
    let a = &1;
    ptr::eq(a, a);

    let a = Rc::new(1);
    Rc::ptr_eq(&a, &a);

    let a = Arc::new(1);
    Arc::ptr_eq(&a, &a);

    let a: &[u8] = b"";
    ptr::eq(a, a);
}
