//@ check-pass

#![warn(unused)]

#[expect(unused_variables)]
fn check_specific_lint() {
    let x = 2;
}

#[expect(unused)]
fn check_lint_group() {
    let x = 15;
}

#[expect(unused_variables)]
fn check_multiple_lint_emissions() {
    let r = 1;
    let u = 8;
    let s = 2;
    let t = 9;
}

mod check_fulfilled_expect_in_macro {
    macro_rules! expect_inside_macro {
        () => {
            #[expect(unused_variables)]
            let x = 0;
        };
    }

    pub fn check_macro() {
        expect_inside_macro!();
    }
}

fn main() {
    check_specific_lint();
    check_lint_group();
    check_multiple_lint_emissions();

    check_fulfilled_expect_in_macro::check_macro();
}
