macro_rules! foo {
    ($e:expr) => {
        #[$e]
        //~^ ERROR expected identifier, found metavariable
        fn foo() {}
    };
}
foo!(inline);

macro_rules! bar {
    ($e:expr) => {
        #[inline($e)]
        //~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `expr` metavariable
        fn bar() {}
    };
}
bar!(always);

fn main() {}
