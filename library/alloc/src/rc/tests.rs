use super::*;

#[test]
fn is_unique() {
    let x = Rc::new(3);
    assert!(Rc::is_unique(&x));
    let y = x.clone();
    assert!(!Rc::is_unique(&x));
    drop(y);
    assert!(Rc::is_unique(&x));
    let w = Rc::downgrade(&x);
    assert!(!Rc::is_unique(&x));
    drop(w);
    assert!(Rc::is_unique(&x));
}
