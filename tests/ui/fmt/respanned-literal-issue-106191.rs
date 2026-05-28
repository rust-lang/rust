//@ proc-macro: format-string-proc-macro.rs

extern crate format_string_proc_macro;

fn main() {
    format_string_proc_macro::respan_to_invalid_format_literal!("¡");
    //~^ ERROR invalid format string: expected `}` but string was terminated
    format_args!(r#concat!("¡        {"));
    //~^ ERROR invalid format string: expected `}` but string was terminated
}
