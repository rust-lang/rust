// ignore-tidy-file-linelength
//@ revisions: stack-protector-2024 stack-protector-2024-allow-deny-reset-by-mitigation stack-protector-2024-deny-reset-by-mitigation
//@ check-fail
//@ compile-flags: -D future-incompatible
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition: 2024

// msvc has an extra unwind dependency of std, normalize it in the error messages
//@ normalize-stderr: "\b(unwind|libc)\b" -> "unwind/libc"

// check that in edition 2024, it is still possible to explicitly
// disallow partial mitigations (in edition=future, they are
// disallowed by default)

// check that explicit deny of stack-protector works in edition 2024
//@ [stack-protector-2024-deny-reset-by-mitigation] compile-flags: -Z deny-partial-mitigations=stack-protector -Z stack-protector=all

// check that this is the case even if there was an "allow" then a "deny"
//@ [stack-protector-2024-allow-deny-reset-by-mitigation] compile-flags: -Z unstable-options -Z allow-partial-mitigations=stack-protector -Z deny-partial-mitigations=stack-protector -Z stack-protector=all

// check that stack-protector is partial-denied in edition 2024
//@ [stack-protector-2024] compile-flags: -Z stack-protector=all

fn main() {}
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
//~? WARN this was previously accepted
//~? WARN this was previously accepted
//~? WARN this was previously accepted
//~? WARN this was previously accepted
//~? WARN this was previously accepted
