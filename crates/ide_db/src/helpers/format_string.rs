//! Tools to work with format string literals for the `format_args!` family of macros.
use syntax::{ast, AstNode, AstToken};

pub fn is_format_string(string: &ast::String) -> bool {
    // Check if `string` is a format string argument of a macro invocation.
    // `string` is a string literal, mapped down into the innermost macro expansion.
    // Since `format_args!` etc. remove the format string when expanding, but place all arguments
    // in the expanded output, we know that the string token is (part of) the format string if it
    // appears in `format_args!` (otherwise it would have been mapped down further).
    //
    // This setup lets us correctly highlight the components of `concat!("{}", "bla")` format
    // strings. It still fails for `concat!("{", "}")`, but that is rare.

    (|| {
        let macro_call = string.syntax().ancestors().find_map(ast::MacroCall::cast)?;
        let name = macro_call.path()?.segment()?.name_ref()?;

        if !matches!(
            name.text().as_str(),
            "format_args" | "format_args_nl" | "const_format_args" | "panic_2015" | "panic_2021"
        ) {
            return None;
        }

        // NB: we match against `panic_2015`/`panic_2021` here because they have a special-cased arm for
        // `"{}"`, which otherwise wouldn't get highlighted.

        Some(())
    })()
    .is_some()
}
