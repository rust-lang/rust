// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is meant as a test case for Issue 3961.
//
// Test via: rustc --pretty normal src/test/pretty/block-comment-wchar.rs
//
// pp-exact:block-comment-wchar.pp
fn f() {
    fn nested() {
        /*
          Spaced2
        */
        /*
          Spaced10
        */
        /*
          Tabbed8+2
        */
        /*
          CR8+2
        */
    }
    /*
      Spaced2:                       (prefixed so start of space aligns with comment)
    */
    /*
    		Tabbed2: (more indented b/c *start* of space will align with comment)
    */
    /*
      Spaced6:                       (Alignment removed and realigning spaces inserted)
    */
    /*
      Tabbed4+2:                     (Alignment removed and realigning spaces inserted)
    */

    /*
      VT4+2:                         (should align)
    */
    /*
      FF4+2:                         (should align)
    */
    /*
      CR4+2:                         (should align)
    */
    /*
      NEL4+2:                        (should align)
    */
    /*
      Ogham Space Mark 4+2:          (should align)
    */
    /*
      Ogham Space Mark 4+2: (should align)
    */
    /*
      Four-per-em space 4+2:         (should align)
    */

    /*
      Ogham Space Mark   count 1: (should align)
      Ogham Space Mark   count 2: (should align)
      Ogham Space Mark   count 3: (should align)
      Ogham Space Mark   count 4: (should align)
      Ogham Space Mark   count 5: (should align)
      Ogham Space Mark   count 6: (should align)
      Ogham Space Mark   count 7: (should align)
      Ogham Space Mark   count 8: (should align)
      Ogham Space Mark   count 9: (should align)
      Ogham Space Mark   count A: (should align)
      Ogham Space Mark   count B: (should align)
      Ogham Space Mark   count C: (should align)
      Ogham Space Mark   count D: (should align)
      Ogham Space Mark   count E: (should align)
      Ogham Space Mark   count F: (should align)
    */



    /* */

    /*
      Hello from offset 6
      Space 6+2:                     compare A
      Ogham Space Mark 6+2: compare B
    */

    /* */

    /*
      Hello from another offset 6 with wchars establishing column offset
      Space 6+2:                     compare C
      Ogham Space Mark 6+2: compare D
    */
}

fn main() {
    // Taken from http://www.unicode.org/Public/UNIDATA/PropList.txt
    let chars =
        ['\x0A', '\x0B', '\x0C', '\x0D', '\x20', '\x85', '\xA0', '\u1680',
         '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006',
         '\u2007', '\u2008', '\u2009', '\u200A', '\u2028', '\u2029', '\u202F',
         '\u205F', '\u3000'];
    for c in chars.iter() {
        let ws = c.is_whitespace();
        println!("{:?} {:?}" , c , ws);
    }
}
