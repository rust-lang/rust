use super::Error;
use crate::fmt;

#[derive(Debug, PartialEq)]
struct A;
#[derive(Debug, PartialEq)]
struct B;

impl fmt::Display for A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "A")
    }
}
impl fmt::Display for B {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B")
    }
}

impl Error for A {}
impl Error for B {}

#[test]
fn downcasting() {
    let mut a = A;
    let a = &mut a as &mut (dyn Error + 'static);
    assert_eq!(a.downcast_ref::<A>(), Some(&A));
    assert_eq!(a.downcast_ref::<B>(), None);
    assert_eq!(a.downcast_mut::<A>(), Some(&mut A));
    assert_eq!(a.downcast_mut::<B>(), None);

    let a: Box<dyn Error> = Box::new(A);
    match a.downcast::<B>() {
        Ok(..) => panic!("expected error"),
        Err(e) => assert_eq!(*e.downcast::<A>().unwrap(), A),
    }
}
