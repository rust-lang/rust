// check-pass
// Allowing the code lint should work without warning and
// the text flow char in the comment should be ignored.

#![allow(text_direction_codepoint_in_comment)]

fn main() {
    // U+2066 LEFT-TO-RIGHT ISOLATE follows:⁦⁦
}
