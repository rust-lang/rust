// edition: 2021
// revisions: cfg no

// [no] check-pass
// Since we're not adding new syntax, `cfg`'d out RTN must pass.


trait Trait {
    #[allow(async_fn_in_trait)]
    async fn m();
}

#[cfg(cfg)]
fn foo<T: Trait<m(): Send>>() {}
//[cfg]~^ ERROR return type notation is experimental
//[cfg]~| ERROR parenthesized generic arguments cannot be used in associated type constraints
//[cfg]~| ERROR associated type `m` not found for `Trait`
//[no]~^^^^ WARN return type notation is experimental
//[no]~| WARN unstable syntax can change at any point in the future, causing a hard error!

fn main() {}
