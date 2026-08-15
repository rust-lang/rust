#![feature(coverage_attribute)]
//@ edition: 2021
//@ reference: attributes.coverage.nesting

// Checks that `#[coverage(..)]` can be applied to modules, and is inherited
// by any enclosed functions.

#[coverage(off)]
mod off {
    fn inherit() {}

    #[coverage(on)]
    fn on() {}

    #[coverage(off)]
    fn off() {}
}

#[coverage(on)]
mod on {
    fn inherit() {}

    #[coverage(on)]
    fn on() {}

    #[coverage(off)]
    fn off() {}
}

#[coverage(off)]
mod nested_a {
    mod nested_b {
        fn inner() {}
    }
}

#[coverage(off)]
fn main() {}
