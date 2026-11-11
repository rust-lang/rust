// This test checks that when no features are passed by Cargo we
// suggest adding some in the Cargo.toml instead of vomitting a
// list of all the expected names
//
//@ check-pass
//@ no-auto-check-cfg
//@ revisions: some none
//@ rustc-env:CARGO_CRATE_NAME=foo
//@ compile-flags: --check-cfg=cfg(docsrs,test)
//@ [none]compile-flags: --check-cfg=cfg(feature,values())
//@ [some]compile-flags: --check-cfg=cfg(feature,values("bitcode"))
//@ [some]compile-flags: --check-cfg=cfg(CONFIG_NVME,values("y"))
//@ dont-require-annotations: HELP

#[cfg(feature = "serde")]
//~^ WARNING unexpected `cfg` condition value
fn ser() {}

#[cfg(feature)]
//~^ WARNING unexpected `cfg` condition value
fn feat() {}

#[cfg(tokio_unstable)]
//~^ WARNING unexpected `cfg` condition name
fn tokio() {}

#[cfg(CONFIG_NVME = "m")]
//[none]~^ WARNING unexpected `cfg` condition name
//[some]~^^ WARNING unexpected `cfg` condition value
//[none]~| HELP Cargo.toml
fn tokio() {}

fn main() {}
