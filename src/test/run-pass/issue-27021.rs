fn main() {
    let mut c = (1, (1, "".to_owned()));
    match c {
        c2 => { (c.1).0 = 2; assert_eq!((c2.1).0, 1); }
    }

    let mut c = (1, (1, (1, "".to_owned())));
    match c.1 {
        c2 => { ((c.1).1).0 = 3; assert_eq!((c2.1).0, 1); }
    }
}
