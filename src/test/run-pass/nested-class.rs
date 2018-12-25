#![allow(non_camel_case_types)]

pub fn main() {
    struct b {
        i: isize,
    }

    impl b {
        fn do_stuff(&self) -> isize { return 37; }
    }

    fn b(i:isize) -> b {
        b {
            i: i
        }
    }

    //  fn b(x:isize) -> isize { panic!(); }

    let z = b(42);
    assert_eq!(z.i, 42);
    assert_eq!(z.do_stuff(), 37);
}
