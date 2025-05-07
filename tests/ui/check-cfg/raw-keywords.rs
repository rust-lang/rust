// This test check that using raw keywords works with --cfg and --check-cfg
// and that the diagnostics suggestions are coherent
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --cfg=true --cfg=async --check-cfg=cfg(r#true,r#async,edition2015,edition2021)
//
//@ revisions: edition2015 edition2021
//@ [edition2015] edition: 2015
//@ [edition2021] edition: 2021

#[cfg(r#true)]
fn foo() {}

#[cfg(tru)]
//~^ WARNING unexpected `cfg` condition name: `tru`
//~^^ SUGGESTION r#true
fn foo() {}

#[cfg(r#false)]
//~^ WARNING unexpected `cfg` condition name: `r#false`
fn foo() {}

#[cfg_attr(edition2015, cfg(async))]
#[cfg_attr(edition2021, cfg(r#async))]
fn bar() {}

#[cfg_attr(edition2015, cfg(await))]
#[cfg_attr(edition2021, cfg(r#await))]
//[edition2015]~^^ WARNING unexpected `cfg` condition name: `await`
//[edition2021]~^^ WARNING unexpected `cfg` condition name: `r#await`
fn zoo() {}

#[cfg(r#raw)]
//~^ WARNING unexpected `cfg` condition name: `raw`
fn foo() {}

fn main() {
    foo();
    bar();
}
