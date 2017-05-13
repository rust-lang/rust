// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[test]
fn test_format_float() {
    assert!("1" == format!("{:.0}", 1.0f64));
    assert!("9" == format!("{:.0}", 9.4f64));
    assert!("10" == format!("{:.0}", 9.9f64));
    assert!("9.8" == format!("{:.1}", 9.849f64));
    assert!("9.9" == format!("{:.1}", 9.851f64));
    assert!("1" == format!("{:.0}", 0.5f64));
}
