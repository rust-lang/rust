//@ rustfmt-style_edition: 2024
//@ rustfmt-wrap_comments: false

fn main() {} /* first
    second
      third */

fn long_first_line() {} /* this is a long first line that exceeds 100 columns and should remain on the same line
hello
  world */
