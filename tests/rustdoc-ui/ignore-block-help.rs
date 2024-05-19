//@ check-pass

/// ```ignore (to-prevent-tidy-error)
/// let heart = '❤️';
/// ```
//~^^^ WARNING could not parse code block
//~| NOTE on by default
//~| NOTE character literal may only contain one codepoint
//~| HELP `ignore` code blocks require valid Rust code
pub struct X;
