//@ known-bug: rust-lang/rust#146754
//@ edition:2021
unsafe extern "C" {
    async fn function() -> [(); || {}];
}
