// check-pass

/// ```ignore (to-prevent-tidy-error)
/// let unterminated = '
/// ```
//~^^^ WARNING could not parse code block
//~| NOTE on by default
//~| NOTE unterminated character literal
//~| HELP `ignore` code blocks require valid Rust code
pub struct X;
