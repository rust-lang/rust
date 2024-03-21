//@ build-pass
//@ ignore-pass (different metadata emitted in different modes)
//@ compile-flags: --json=diagnostic-short,artifacts --error-format=json
//@ normalize-stderr-test: "json_options\.[0-9a-zA-Z]+-cgu" -> "json_options.HASH-cgu"

#![crate_type = "lib"]
