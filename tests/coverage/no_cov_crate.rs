#![feature(coverage_attribute)]
// Enables `coverage(off)` on the entire crate
//@ reference: attributes.coverage.intro
//@ reference: attributes.coverage.nesting

#[coverage(off)]
fn do_not_add_coverage_1() {
    println!("called but not covered");
}

fn do_not_add_coverage_2() {
    #![coverage(off)]
    println!("called but not covered");
}

#[coverage(off)]
#[allow(dead_code)]
fn do_not_add_coverage_not_called() {
    println!("not called and not covered");
}

fn add_coverage_1() {
    println!("called and covered");
}

fn add_coverage_2() {
    println!("called and covered");
}

#[allow(dead_code)]
fn add_coverage_not_called() {
    println!("not called but covered");
}

// FIXME: These test-cases illustrate confusing results of nested functions.
// See https://github.com/rust-lang/rust/issues/93319
mod nested_fns {
    #[coverage(off)]
    pub fn outer_not_covered(is_true: bool) {
        fn inner(is_true: bool) {
            if is_true {
                println!("called and covered");
            } else {
                println!("absolutely not covered");
            }
        }
        println!("called but not covered");
        inner(is_true);
    }

    pub fn outer(is_true: bool) {
        println!("called and covered");
        inner_not_covered(is_true);

        #[coverage(off)]
        fn inner_not_covered(is_true: bool) {
            if is_true {
                println!("called but not covered");
            } else {
                println!("absolutely not covered");
            }
        }
    }

    pub fn outer_both_covered(is_true: bool) {
        println!("called and covered");
        inner(is_true);

        fn inner(is_true: bool) {
            if is_true {
                println!("called and covered");
            } else {
                println!("absolutely not covered");
            }
        }
    }
}

fn main() {
    let is_true = std::env::args().len() == 1;

    do_not_add_coverage_1();
    do_not_add_coverage_2();
    add_coverage_1();
    add_coverage_2();

    nested_fns::outer_not_covered(is_true);
    nested_fns::outer(is_true);
    nested_fns::outer_both_covered(is_true);
}
