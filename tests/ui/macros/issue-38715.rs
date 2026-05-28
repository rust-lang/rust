#[macro_export]
macro_rules! foo { () => {} }

#[macro_export]
macro_rules! foo { () => {} } //~ ERROR the name `foo` is defined multiple times

mod inner1 {
    #[macro_export]
    macro_rules! bar { () => {} }
}

mod inner2 {
    #[macro_export]
    macro_rules! bar { () => {} } //~ ERROR the name `bar` is defined multiple times
}

fn main() {}
