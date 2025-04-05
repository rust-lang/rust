#![feature(contracts)]
//~^ WARNING the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]
//~| ERROR `main` function not found in crate `attribute_affected_trait_bound_issue_137129` [E0601]
#![core::contracts::ensures]
//~^ ERROR inner macro attributes are unstable [E0658]
//~| ERROR contract annotations can only be used on functions
struct A {
    b: dyn A + 'static,
}
fn f1() {}
