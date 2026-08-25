//@ proc-macro: format-string-proc-macro.rs

#[macro_use]
extern crate format_string_proc_macro;

macro_rules! def_site {
    () => { "{foo}" } //~ ERROR: there is no argument named `foo`
}

macro_rules! call_site {
    ($fmt:literal) => { $fmt }
}

fn main() {
    format!(concat!("{foo}"));         //~ ERROR: there is no argument named `foo`
    format!(concat!("{ba", "r} {}"), 1);     //~ ERROR: there is no argument named `bar`

    format!(def_site!());
    format!(call_site!("{foo}")); //~ ERROR: there is no argument named `foo`

    format!(foo_with_input_span!("")); //~ ERROR: there is no argument named `foo`
}
