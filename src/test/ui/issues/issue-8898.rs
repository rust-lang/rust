// run-pass

fn assert_repr_eq<T: std::fmt::Debug>(obj : T, expected : String) {
    assert_eq!(expected, format!("{:?}", obj));
}

pub fn main() {
    let abc = [1, 2, 3];
    let tf = [true, false];
    let x  = [(), ()];
    let slice = &x[..1];

    assert_repr_eq(&abc[..], "[1, 2, 3]".to_string());
    assert_repr_eq(&tf[..], "[true, false]".to_string());
    assert_repr_eq(&x[..], "[(), ()]".to_string());
    assert_repr_eq(slice, "[()]".to_string());
    assert_repr_eq(&x[..], "[(), ()]".to_string());
}
