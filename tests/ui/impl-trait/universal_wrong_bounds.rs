use std::fmt::Display;

fn foo(f: impl Display + Clone) -> String {
    wants_debug(f);
    wants_display(f);
    wants_clone(f);
}

fn wants_debug(g: impl Debug) { } //~ ERROR cannot find trait `Debug` in this scope
fn wants_display(g: impl Debug) { } //~ ERROR cannot find trait `Debug` in this scope
fn wants_clone(g: impl Clone) { }

fn main() {}
