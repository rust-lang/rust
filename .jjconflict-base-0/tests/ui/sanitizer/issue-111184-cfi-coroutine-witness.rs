// Regression test for issue 111184, where ty::CoroutineWitness were not expected to occur in
// encode_ty and caused the compiler to ICE.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Zsanitizer=cfi -C unsafe-allow-abi-mismatch=sanitizer
//@ edition: 2021
//@ no-prefer-dynamic
//@ only-x86_64-unknown-linux-gnu
//@ build-pass
//@ ignore-backends: gcc

use std::future::Future;

async fn foo() {}
fn bar<T>(_: impl Future<Output = T>) {}

fn main() {
    bar(foo());
}
