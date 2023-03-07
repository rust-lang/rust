use std::fmt::Display;

fn foo(f: impl Display + Clone) -> String {
    wants_debug(f);
    wants_display(f);
    wants_clone(f);
}

fn wants_debug(g: impl Debug) { } //~ ERROR expected trait, found derive macro `Debug`
fn wants_display(g: impl Debug) { } //~ ERROR expected trait, found derive macro `Debug`
fn wants_clone(g: impl Clone) { }

fn main() {}
