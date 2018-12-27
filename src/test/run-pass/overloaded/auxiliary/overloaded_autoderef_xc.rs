use std::ops::Deref;

struct DerefWithHelper<H, T> {
    pub helper: H,
    pub value: Option<T>
}

trait Helper<T> {
    fn helper_borrow(&self) -> &T;
}

impl<T> Helper<T> for Option<T> {
    fn helper_borrow(&self) -> &T {
        self.as_ref().unwrap()
    }
}

impl<T, H: Helper<T>> Deref for DerefWithHelper<H, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.helper.helper_borrow()
    }
}

// Test cross-crate autoderef + vtable.
pub fn check<T: PartialEq>(x: T, y: T) -> bool {
    let d: DerefWithHelper<Option<T>, T> = DerefWithHelper { helper: Some(x), value: None };
    d.eq(&y)
}
