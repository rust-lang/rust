//@ proc-macro: format-string-proc-macro.rs
//@ check-pass

extern crate format_string_proc_macro;

fn main() {
    let a = 0;
    format_string_proc_macro::capture_a_with_prepended_space_preserve_span!("{a}");
}
