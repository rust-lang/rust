#![deny(rustdoc::invalid_markdown_table)]

//! | col1 |
//! | ---- |
//! | `code_with(|arg| arg)` |
//~^ ERROR invalid_markdown_table
//! | one `|` b |
//~^ ERROR invalid_markdown_table
//!
// Testing another lint emission on the same doc comment.
//!
//! | col1 |
//! | ---- |
//! | `code_with(|arg| arg)` |
//~^ ERROR invalid_markdown_table

// We check that the extra whitespace characters at the end of the line won't trigger
// the lint.
mod b {
    //! | col |
    //! | ---- |
    #![doc = "| code_with |            "]
}

// We check that the `\|` is correctly handled as well (ie not emitting the lint).
mod c {
    //! | col |
    //! | ---- |
    //! | a \| still same cell |
    //!
    //! now with double-backslashes;
    //! yes, it really does work this way,
    //! but it's *weird* compared to what
    //! you would naively expect
    //!
    //! | col |
    //! | ---- |
    //! | a \\| still same cell |
}

// We check that content after a table row also emits.
mod d {
    //! | col |
    //! | ---- |
    //! | code_with | aaaaa
    //~^ ERROR invalid_markdown_table
    //! blob
    //!
    //! | col |
    //! | ---- |
    //! | code_with | \|
    //~^ ERROR invalid_markdown_table
    //! blob
    //!
    //! | col |
    //! | ---- |
    //! | code_with | \\|
    //~^ ERROR invalid_markdown_table
    //! blob
    //!
    //! one | two
    //! -|-
    //! a | b | c
    //~^ ERROR invalid_markdown_table
    //! a |
}

// More cases that are ignored.
mod e {
    //! | one | two |
    //! |-|-|
    //! | a |
    //! | b
}

// Weird corner case where the table ends with a pipe,
// but doesn't start with it
mod f {
    //! one |
    //! ----|
    //!   a | b | c
    //~^ ERROR invalid_markdown_table
    //!   a | b |
    //~^ ERROR invalid_markdown_table
    //!   a | b
    //~^ ERROR invalid_markdown_table
    //!   a |
}

// Weird corner case where the table ends with a pipe,
// but doesn't start with it
mod g {
    //! | one
    //! |----
    //! |  a | b | c
    //~^ ERROR invalid_markdown_table
    //! |  a | b |
    //~^ ERROR invalid_markdown_table
    //! |  a | b
    //~^ ERROR invalid_markdown_table
    //! |  a |
    //! |  a
}
