// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub fn main() {
    enum t { t1(isize), t2(isize), }

    let _x: Box<_> = box t::t1(10);

    /*alt *x {
      t1(a) {
        assert_eq!(a, 10);
      }
      _ { panic!(); }
    }*/

    /*alt x {
      box t1(a) {
        assert_eq!(a, 10);
      }
      _ { panic!(); }
    }*/
}
