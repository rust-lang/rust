// rustfmt-wrap_comments: true
// rustfmt-comment_width: 100

fn f() {
    // Note: off-by-one bug: the 100 width wide line should not be formatted
    // https://github.com/rust-lang/rustfmt/issues/7167
    foo(
        // this comment is 100 characters wide! The last word should fit on the following line --
        // aa This line is 92 characters wide -------------------------------------------------
        1,
    );

    // this comment should be left alone
    // this comment is 100 characters wide! --------------------------------------------------------

    // this comment is 101 characters wide! The last word should fit on the following line -------
    // aa This line is 92 characters wide ----------------------------------------------------
}
