// run-pass

pub trait Clone2 {
    /// Returns a copy of the value. The contents of boxes
    /// are copied to maintain uniqueness, while the contents of
    /// managed pointers are not copied.
    fn clone(&self) -> Self;
}

trait Getter<T: Clone> {
    fn do_get(&self) -> T;

    fn do_get2(&self) -> (T, T) {
        let x = self.do_get();
        (x.clone(), x.clone())
    }

}

impl Getter<isize> for isize {
    fn do_get(&self) -> isize { *self }
}

impl<T: Clone> Getter<T> for Option<T> {
    fn do_get(&self) -> T { self.as_ref().unwrap().clone() }
}


pub fn main() {
    assert_eq!(3.do_get2(), (3, 3));
    assert_eq!(Some("hi".to_string()).do_get2(), ("hi".to_string(), "hi".to_string()));
}
