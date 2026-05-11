// Specified by both C and Rust
pub const SINGLE_QUOTE: char = '\'';
pub const DOUBLE_QUOTE: char = '\"';
pub const BACKSLASH: char = '\\';
pub const NEWLINE: char = '\n';
pub const CARRIAGE_RETURN: char = '\r';
pub const HORIZONTAL_TAB: char = '\t';
pub const NULL: char = '\0';

// Specified by C, but not Rust
pub const QUESTION_MARK: char = '\?'; //~ ERROR unknown character escape
pub const AUDIBLE_BELL: char = '\a'; //~ ERROR unknown character escape
pub const BACKSPACE: char = '\b'; //~ ERROR unknown character escape
pub const FORM_FEED: char = '\f'; //~ ERROR unknown character escape
pub const VERTICAL_TAB: char = '\v'; //~ ERROR unknown character escape
pub const OCTAL: char = '\1'; //~ ERROR unknown character escape

// Not specified by C, but recognized by GCC as an extension.
// Used for ANSI escape sequences in terminal emulators.
pub const ESCAPE: char = '\e'; //~ ERROR unknown character escape

fn main() {}
