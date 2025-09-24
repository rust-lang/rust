macro_rules! foo {
    ($e:expr) => {
        #[cfg_attr(true, $e)]
        //~^ ERROR expected identifier, found metavariable
        fn foo() {}
    }
}

foo!(inline);

fn main() {}
