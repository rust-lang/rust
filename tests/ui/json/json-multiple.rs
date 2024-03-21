//@ build-pass
//@ ignore-pass (different metadata emitted in different modes)
//@ compile-flags: --json=diagnostic-short --json artifacts --error-format=json
//@ normalize-stderr-test: "json_multiple\.[0-9a-zA-Z]+-cgu" -> "json_multiple.HASH-cgu"

#![crate_type = "lib"]
