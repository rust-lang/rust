//@ run-pass

struct NotCopy<'a>(&'a mut bool);

fn test<'a>(m: &'a mut bool) {
    let r = (NotCopy(m),);
    drop(r.0);

    // `r.0` is now moved  out of,
    // but `r` is not fully moved

    let _ = r.0.0;
    let _: &mut _ = r.0.0;
    let _: &mut bool = r.0.0;
    let _: &'a mut bool = r.0.0;
}

fn main() {
    test(&mut false);
}
