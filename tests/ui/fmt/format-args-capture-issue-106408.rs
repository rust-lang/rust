// check-pass
// aux-build:format-string-proc-macro.rs

extern crate format_string_proc_macro;

fn main() {
    // While literal macros like `format_args!(concat!())` are not supposed to work with implicit
    // captures, it should work if the whole invocation comes from a macro expansion (#106408).
    format_string_proc_macro::format_args_captures!();
}
