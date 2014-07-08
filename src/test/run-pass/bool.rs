// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Basic boolean tests

fn main() {
    assert_eq!(false.eq(&true), false);
    assert_eq!(false == false, true);
    assert_eq!(false != true, true);
    assert_eq!(false.ne(&false), false);

    assert_eq!(false.bitand(&false), false);
    assert_eq!(true.bitand(&false), false);
    assert_eq!(false.bitand(&true), false);
    assert_eq!(true.bitand(&true), true);

    assert_eq!(false & false, false);
    assert_eq!(true & false, false);
    assert_eq!(false & true, false);
    assert_eq!(true & true, true);

    assert_eq!(false.bitor(&false), false);
    assert_eq!(true.bitor(&false), true);
    assert_eq!(false.bitor(&true), true);
    assert_eq!(true.bitor(&true), true);

    assert_eq!(false | false, false);
    assert_eq!(true | false, true);
    assert_eq!(false | true, true);
    assert_eq!(true | true, true);

    assert_eq!(false.bitxor(&false), false);
    assert_eq!(true.bitxor(&false), true);
    assert_eq!(false.bitxor(&true), true);
    assert_eq!(true.bitxor(&true), false);

    assert_eq!(false ^ false, false);
    assert_eq!(true ^ false, true);
    assert_eq!(false ^ true, true);
    assert_eq!(true ^ true, false);

    assert_eq!(!true, false);
    assert_eq!(!false, true);

    let s = false.to_string();
    assert_eq!(s.as_slice(), "false");
    let s = true.to_string();
    assert_eq!(s.as_slice(), "true");

    assert!(true > false);
    assert!(!(false > true));

    assert!(false < true);
    assert!(!(true < false));

    assert!(false <= false);
    assert!(false >= false);
    assert!(true <= true);
    assert!(true >= true);

    assert!(false <= true);
    assert!(!(false >= true));
    assert!(true >= false);
    assert!(!(true <= false));

    assert!(true.cmp(&true) == Equal);
    assert!(false.cmp(&false) == Equal);
    assert!(true.cmp(&false) == Greater);
    assert!(false.cmp(&true) == Less);
}
