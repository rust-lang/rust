// run-fail
// check-run-results

// exec-env:RUST_BACKTRACE=full
// revisions: with-remap without-remap
// compile-flags: -g -Ztranslate-remapped-path-to-local-path=yes
// [with-remap]compile-flags: --remap-path-prefix={{rust-src-base}}=remapped
// [without-remap]compile-flags:

fn main() {
    Vec::<String>::with_capacity(!0);
}
