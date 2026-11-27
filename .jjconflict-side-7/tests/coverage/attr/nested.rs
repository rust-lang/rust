#![feature(coverage_attribute, stmt_expr_attributes)]
//@ edition: 2021
//@ reference: attributes.coverage.nesting

// Demonstrates the interaction between #[coverage(off)] and various kinds of
// nested function.

#[coverage(off)]
fn do_stuff() {}

#[coverage(off)]
fn outer_fn() {
    fn middle_fn() {
        fn inner_fn() {
            do_stuff();
        }
        do_stuff();
    }
    do_stuff();
}

struct MyOuter;
impl MyOuter {
    #[coverage(off)]
    fn outer_method(&self) {
        struct MyMiddle;
        impl MyMiddle {
            fn middle_method(&self) {
                struct MyInner;
                impl MyInner {
                    fn inner_method(&self) {
                        do_stuff();
                    }
                }
                do_stuff();
            }
        }
        do_stuff();
    }
}

trait MyTrait {
    fn trait_method(&self);
}
impl MyTrait for MyOuter {
    #[coverage(off)]
    fn trait_method(&self) {
        struct MyMiddle;
        impl MyTrait for MyMiddle {
            fn trait_method(&self) {
                struct MyInner;
                impl MyTrait for MyInner {
                    fn trait_method(&self) {
                        do_stuff();
                    }
                }
                do_stuff();
            }
        }
        do_stuff();
    }
}

fn closure_expr() {
    let _outer = #[coverage(off)]
    || {
        let _middle = || {
            let _inner = || {
                do_stuff();
            };
            do_stuff();
        };
        do_stuff();
    };
    do_stuff();
}

// This syntax is allowed, even without #![feature(stmt_expr_attributes)].
fn closure_tail() {
    let _outer = {
        #[coverage(off)]
        || {
            let _middle = {
                || {
                    let _inner = {
                        || {
                            do_stuff();
                        }
                    };
                    do_stuff();
                }
            };
            do_stuff();
        }
    };
    do_stuff();
}

#[coverage(off)]
fn main() {
    outer_fn();
    MyOuter.outer_method();
    MyOuter.trait_method();
    closure_expr();
    closure_tail();
}
