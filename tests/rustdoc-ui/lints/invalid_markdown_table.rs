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
}

// We check that content after a table row also emits.
mod d {
    //! | col |
    //! | ---- |
    //! | code_with | aaaaa
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
