#![deny(rustdoc::unescaped_pipe_in_table_cell)]

//! | col1 |
//! | ---- |
//! | `code_with(|arg| arg)` |
//~^ ERROR unescaped_pipe_in_table_cell
//! | one `|` b |
//~^ ERROR unescaped_pipe_in_table_cell
//!
// Testing another lint emission on the same doc comment.
//!
//! | col1 |
//! | ---- |
//! | `code_with(|arg| arg)` |
//~^ ERROR unescaped_pipe_in_table_cell

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

// We check that content after a table row is ignored.
mod d {
    //! | col |
    //! | ---- |
    //! | code_with | aaaaa
    //^ the "aaaaa" part will be ignored.
}
