//! We try to hide the internals of builtin-like macros from error messages,
//! but *not* if `-Z macro-backtrace` is enabled

//@ revisions: default with
//@[with] compile-flags: -Z macro-backtrace
//@[with] error-pattern: in this expansion of `$crate::format_args_nl!`

fn main(){
    let x: u32;

    println!("{x}");
    //~^ERROR used binding `x` isn't initialized
}
