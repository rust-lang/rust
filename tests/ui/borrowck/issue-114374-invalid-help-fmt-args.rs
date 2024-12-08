#![allow(dead_code)]

fn bar<'a>(_: std::fmt::Arguments<'a>) {}
fn main() {
    let x = format_args!("a {} {} {}.", 1, format_args!("b{}!", 2), 3);
    //~^ ERROR temporary value dropped while borrowed

    bar(x);

    let foo = format_args!("{}", "hi");
    //~^ ERROR temporary value dropped while borrowed
    bar(foo);

    let foo = format_args!("hi"); // no placeholder in arguments, so no error
    bar(foo);
}
