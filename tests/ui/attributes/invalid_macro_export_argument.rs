#[macro_export(hello, world)]
//~^ ERROR malformed `macro_export` attribute input
//~| HELP try changing it to one of the following valid forms of the attribute
macro_rules! a {
    () => ()
}

#[macro_export(not_local_inner_macros)]
//~^ ERROR malformed `macro_export` attribute input
//~| HELP try changing it to one of the following valid forms of the attribute
macro_rules! b {
    () => ()
}

#[macro_export]
macro_rules! c {
    () => ()
}
#[macro_export(local_inner_macros)]
macro_rules! d {
    () => ()
}

#[macro_export()]
//~^ ERROR malformed `macro_export` attribute input
//~| HELP try changing it to one of the following valid forms of the attribute
macro_rules! e {
    () => ()
}

#[macro_export("blah")]
//~^ ERROR malformed `macro_export` attribute input
//~| HELP try changing it to one of the following valid forms of the attribute
macro_rules! f {
    () => ()
}

fn main() {}
