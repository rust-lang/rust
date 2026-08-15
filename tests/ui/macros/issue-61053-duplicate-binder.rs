#![deny(meta_variable_misuse)]

macro_rules! foo {
    () => {};
    (error) => {
        macro_rules! bar {
            ($x:tt $x:tt) => { $x }; //~ ERROR duplicate matcher binding
        }
    };
}

fn main() {
    foo!();
}
