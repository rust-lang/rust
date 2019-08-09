macro_rules! foo {
    () => {
        #[cfg_attr(all(), unknown)]
        //~^ ERROR cannot find attribute macro `unknown` in this scope
        fn foo() {}
    }
}

foo!();

fn main() {}
