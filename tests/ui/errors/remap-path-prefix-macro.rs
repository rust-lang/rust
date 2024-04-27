//@ run-pass
//@ check-run-results

//@ revisions: normal with-macro-scope without-macro-scope
//@ compile-flags: --remap-path-prefix={{src-base}}=remapped
//@ [with-macro-scope]compile-flags: -Zremap-path-scope=macro,diagnostics
//@ [without-macro-scope]compile-flags: -Zremap-path-scope=diagnostics
// no-remap-src-base: Manually remap, so the remapped path remains in .stderr file.

fn main() {
    println!("{}", file!());
}
