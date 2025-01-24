// rustfmt-wrap_comments: true
// rustfmt-style_edition: 2024

// check that a line below max_width does not get over the limit when wrapping
// it in a block comment
fn func() {
    let x = 42;
    /*
    let something = "one line line  line  line  line  line  line  line  line  line  line  line line
  two lines
         three lines";
     */
}
