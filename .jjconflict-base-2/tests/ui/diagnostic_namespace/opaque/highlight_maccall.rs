//@revisions: default -Zmacro-backtrace
//@[-Zmacro-backtrace] compile-flags: -Z macro-backtrace

#![feature(diagnostic_opaque)]

#[diagnostic::opaque]
macro_rules! my_error {
    () => {{
        compile_error!("oh no")
        //~^ ERROR oh no
    }}
}


fn main() {
    my_error!();
}
