//@ revisions: e2015 e2021
//@[e2015] edition:2015
//@[e2021] edition:2021

// Regression test for https://github.com/rust-lang/rust/issues/147595.
// The `unused_variables` typo suggestion must not print a pattern path that
// cannot be resolved from the binding site.

#![deny(unused_variables)]

mod enclosed {
    pub const X: i32 = 1;
}

mod separated {
    pub fn let_else_path_qualification(x: Option<i32>) {
        let Some(x) = x else { return };
        //~^ ERROR unused variable: `x`
        //~| HELP you might have meant to pattern match on the similarly named constant `X`
        //~| HELP if this is intentional, prefix it with an underscore
    }
}

mod utils {
    pub fn simple_binding_with_imported_const() {
        use crate::system;
        let x = system::Y;
        //~^ ERROR unused variable: `x`
        //~| HELP if this is intentional, prefix it with an underscore
    }
}

mod same_module {
    const GOOD: i32 = 0;

    pub fn same_module_const_suggestion(x: Option<i32>) {
        let Some(good) = x else { return };
        //~^ ERROR unused variable: `good`
        //~| HELP you might have meant to pattern match on the similarly named constant `GOOD`
        //~| HELP if this is intentional, prefix it with an underscore
    }
}

fn function_local_const_suggestion(x: Option<i32>) {
    const LOCAL: i32 = 0;
    let Some(local) = x else { return };
    //~^ ERROR unused variable: `local`
    //~| HELP you might have meant to pattern match on the similarly named constant `LOCAL`
    //~| HELP if this is intentional, prefix it with an underscore
}

mod variants {
    pub enum State {
        Ready,
    }
}

mod separated_variant {
    pub fn cross_module_variant_path(x: Option<crate::variants::State>) {
        let Some(ready) = x else { return };
        //~^ ERROR unused variable: `ready`
        //~| HELP you might have meant to pattern match on the similarly named variant `Ready`
        //~| HELP if this is intentional, prefix it with an underscore
    }
}

mod system {
    pub const Y: u32 = 0;
}

// `const _: T = ...` items have no name that can be written in a pattern: a
// suggestion of `path::_` would be syntactically invalid. Such items must be
// excluded from typo suggestions even when their type and name distance would
// otherwise match.
mod anonymous_const {
    pub const _: () = ();
}

fn no_anonymous_const_suggestion(x: Option<()>) {
    let Some(x) = x else { return };
    //~^ ERROR unused variable: `x`
    //~| HELP if this is intentional, prefix it with an underscore
}

// A `const` that is not accessible from the binding site (private to another
// module) must not be offered as a suggestion: applying the rewrite would
// produce an error about a private item, replacing one diagnostic with
// another. The `value` binding deliberately matches the private const's name
// up to case, which is the strongest score in `find_best_match_for_name`.
mod private_const {
    #[allow(dead_code)]
    const VALUE: i32 = 0;
}

fn no_inaccessible_const_suggestion(x: Option<i32>) {
    let Some(value) = x else { return };
    //~^ ERROR unused variable: `value`
    //~| HELP if this is intentional, prefix it with an underscore
}

// A `const` whose name is textually far from the binding (beyond the default
// `find_best_match_for_name` threshold of `max(len, 3) / 3`) must not be
// suggested either: the user did not "almost" mean this constant. The binding
// `xyzzy` is deliberately chosen to be a distance of at least 5 from every
// const in this file.
mod distant_name {
    pub const ENTIRELY_UNRELATED_LONGER_NAME: i32 = 0;
}

fn no_distant_name_suggestion(x: Option<i32>) {
    let Some(xyzzy) = x else { return };
    //~^ ERROR unused variable: `xyzzy`
    //~| HELP if this is intentional, prefix it with an underscore
}

fn main() {}
