// This is meant as a test case for Issue 3961.
//
// Test via: rustc -Zunpretty normal tests/pretty/block-comment-wchar.rs
// ignore-tidy-cr
// ignore-tidy-tab
//@ pp-exact:block-comment-wchar.pp
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
    // Taken from https://www.unicode.org/Public/UNIDATA/PropList.txt
    let chars =
        ['\x0A', '\x0B', '\x0C', '\x0D', '\x20', '\u{85}', '\u{A0}',
                '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}',
                '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}',
                '\u{2009}', '\u{200A}', '\u{2028}', '\u{2029}', '\u{202F}',
                '\u{205F}', '\u{3000}'];
    for c in &chars { let ws = c.is_whitespace(); println!("{} {}", c, ws); }
}
