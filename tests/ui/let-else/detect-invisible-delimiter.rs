// The user shouldn't need to wrap the expression in parentheses(#147899)
//@check-pass
#![allow(irrefutable_let_patterns)]
struct Thing {}
macro_rules! foo {
    ($e:expr) => {
        let _ = $e else {
            return;
        };
    };
}

fn main() {
    foo!(true && true);
    foo!(Thing {});
}
