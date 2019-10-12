// check-pass

#![feature(decl_macro)]

mod single {
    pub trait Single {
        fn single(&self) {}
    }

    impl Single for u8 {}
}
mod glob {
    pub trait Glob {
        fn glob(&self) {}
    }

    impl Glob for u8 {}
}

macro gen_imports() {
    use single::Single;
    use glob::*;
}
gen_imports!();

fn main() {
    0u8.single(); // FIXME, should be an error, `Single` shouldn't be in scope here
    0u8.glob(); // FIXME, should be an error, `Glob` shouldn't be in scope here
}
