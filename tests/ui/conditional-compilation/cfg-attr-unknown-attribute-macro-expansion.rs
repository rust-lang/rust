macro_rules! foo {
    () => {
        #[cfg_attr(all(), unknown)]
        //~^ ERROR cannot find attribute `unknown`
        fn foo() {}
    }
}

foo!();

fn main() {}
