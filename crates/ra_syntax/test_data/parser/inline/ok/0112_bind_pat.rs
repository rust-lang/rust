fn main() {
    let a = ();
    let mut b = ();
    let ref c = ();
    let ref mut d = ();
    let e @ _ = ();
    let ref mut f @ g @ _ = ();
    let box i = Box::new(1i32);
}
