macro_rules! foo {
    () => {
        #[cfg_attr(all(), unknown)] //~ ERROR `unknown` is currently unknown
        fn foo() {}
    }
}

foo!();

fn main() {}
