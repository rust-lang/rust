// Regression test for the issue #44692

macro_rules! hang { () => {
    { //~ ERROR format argument must be a string literal
        #[derive(Clone)]
        struct S;

        ""
    }
}}

fn main() {
    format_args!(hang!());
}
