// check-pass

#![feature(lint_reasons)]

#[expect(unused_variables, reason = "All emissions should be consumed by the nested expect")]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectation]
//~| NOTE #[warn(unfulfilled_lint_expectation)]` on by default
//~| NOTE All emissions should be consumed by the nested expect
mod oof {
    #[expect(unused_variables, reason = "This should collect all unused variable emissions")]
    fn bar() {
        let mut c = 0;
        let mut l = 0;
        let mut l = 0;
        let mut i = 0;
        let mut p = 0;
        let mut y = 0;
    }
}

fn main() {}
