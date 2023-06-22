// edition: 2021
// revisions: cfg_current cfg_next no_current no_next
// [cfg_next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// [no_next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty

// [no_current] check-pass
// [no_next] check-pass
// Since we're not adding new syntax, `cfg`'d out RTN must pass.

#![feature(async_fn_in_trait)]

trait Trait {
    async fn m();
}

#[cfg(any(cfg_current, cfg_next))]
fn foo<T: Trait<m(): Send>>() {}
//[cfg_current]~^ ERROR return type notation is experimental
//[cfg_current]~| ERROR parenthesized generic arguments cannot be used in associated type constraints
//[cfg_current]~| ERROR associated type `m` not found for `Trait`
//[cfg_next]~^^^^ ERROR return type notation is experimental
//[cfg_next]~| ERROR parenthesized generic arguments cannot be used in associated type constraints
//[cfg_next]~| ERROR associated type `m` not found for `Trait`
//[no_current]~^^^^^^^ WARN return type notation is experimental
//[no_current]~| WARN unstable syntax can change at any point in the future, causing a hard error!
//[no_next]~^^^^^^^^^ WARN return type notation is experimental
//[no_next]~| WARN unstable syntax can change at any point in the future, causing a hard error!

fn main() {}
