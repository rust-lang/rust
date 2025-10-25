macro_rules! foo {
    ($e:expr) => {
        #[$e]
        //~^ ERROR expected identifier, found metavariable
        fn foo() {}
    }
}

foo!(inline);

fn main() {}
