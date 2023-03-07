// run-pass

fn main() {
    let x = Box::new(0);
    assert_eq!(0, *x + { drop(x); let _ = Box::new(main); 0 });
}
