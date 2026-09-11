#![allow(dead_code)] // bar

//! Doc comment
fn test() {
// comment
        // comment2

    code(); /* leave this comment alone!
             * ok? */

        /* Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a
         * diam lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam
         * viverra nec consectetur ante hendrerit. Donec et mollis dolor.
         * Praesent et diam eget libero egestas mattis sit amet vitae augue. Nam
         * tincidunt congue enim, ut porta lorem lacinia consectetur. Donec ut
         * libero sed arcu vehicula ultricies a non tortor. Lorem ipsum dolor sit
         * amet, consectetur adipiscing elit. Aenean ut gravida lorem. Ut turpis
         * felis, pulvinar a semper sed, adipiscing id dolor. */

    // Very loooooooooooooooooooooooooooooooooooooooooooooooooooooooong comment that should be split

                    // println!("{:?}", rewrite_comment(subslice,
                    //                                       false,
                    //                                       comment_width,
                    //                                       self.block_indent,
                    //                                       self.config)
                    //                           .unwrap());

    funk(); //dontchangeme
            // or me
}

  /// test123
fn doc_comment() {
}

/*
Regression test for issue #956

(some very important text)
*/

/*
fn debug_function() {
    println!("hello");
}
// */

#[link_section=".vectors"]
#[no_mangle] // Test this attribute is preserved.
#[cfg_attr(rustfmt, rustfmt::skip)]
pub static ISSUE_1284: [i32; 16] = [];
