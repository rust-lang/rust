//@ proc-macro: format-string-proc-macro.rs
//@ ignore-backends: gcc

extern crate format_string_proc_macro;

fn main() {
    format_string_proc_macro::bad_format_args_captures!();
    //~^ ERROR there is no argument named `x`
}
