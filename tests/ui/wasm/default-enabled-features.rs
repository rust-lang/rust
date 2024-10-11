//@ only-wasm32
//@ build-pass

#![feature(wasm_target_feature)]

fn main() {
    #[cfg(any(
        // on-by-default features
        not(target_feature = "multivalue"),
        not(target_feature = "mutable-globals"),
        not(target_feature = "reference-types"),
        not(target_feature = "sign-ext"),

        // off-by-default features
        target_feature = "atomics",
        target_feature = "bulk-memory",
        target_feature = "exception-handling",
        target_feature = "extended-const",
        target_feature = "nontrapping-fptoint",
        target_feature = "simd128",
        target_feature = "relaxed-simd",
    ))]
    compile_error!(
        "\
If this test fails to compile then it means that the default set of features
active on WebAssembly targets is no longer what it used to be. This is likely
due to LLVM being updated and changing the definition of the `generic` CPU in
wasm.

If you'd be so obliged please update the feature listings above this error
message. Additionally please update this file too:

* src/doc/rustc/src/platform-support/wasm32-unknown-unknown.md

Specifically the section about \"Enabled WebAssembly features\". If you'd also
be so kind as to ping wasm maintainers and/or the release team in the PR that
updates this test it would also be much appreciated to ensure that this change
is communicated out to users.
"
    );
}
