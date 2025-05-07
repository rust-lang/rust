//@ revisions: only-rustc cargo-invoked
//@[only-rustc] unset-rustc-env:CARGO_CRATE_NAME
//@[cargo-invoked] rustc-env:CARGO_CRATE_NAME=foo
fn main() {
    let page_size = page_size::get();
    //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `page_size`
    //~| NOTE use of unresolved module or unlinked crate `page_size`
    //[cargo-invoked]~^^^ HELP if you wanted to use a crate named `page_size`, use `cargo add
    //[only-rustc]~^^^^ HELP you might be missing a crate named `page_size`
}
