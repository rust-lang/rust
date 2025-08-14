[package]
name = "rustc_abi"
version = "0.0.0"
edition = "2024"

[dependencies]
# tidy-alphabetical-start
bitflags = "2.4.1"
rand = { version = "0.9.0", default-features = false, optional = true }
rand_xoshiro = { version = "0.7.0", optional = true }
rustc_data_structures = { path = "../rustc_data_structures", optional = true }
rustc_error_messages = { path = "../rustc_error_messages", optional = true }
rustc_hashes = { path = "../rustc_hashes" }
rustc_index = { path = "../rustc_index", default-features = false }
rustc_macros = { path = "../rustc_macros", optional = true }
rustc_serialize = { path = "../rustc_serialize", optional = true }
rustc_span = { path = "../rustc_span", optional = true }
tracing = "0.1"
# tidy-alphabetical-end

[features]
# tidy-alphabetical-start
default = ["nightly", "randomize"]
# rust-analyzer depends on this crate and we therefore require it to built on a stable toolchain
# without depending on rustc_data_structures, rustc_macros and rustc_serialize
nightly = [
    "dep:rustc_data_structures",
    "dep:rustc_error_messages",
    "dep:rustc_macros",
    "dep:rustc_serialize",
    "dep:rustc_span",
    "rustc_index/nightly",
]
randomize = ["dep:rand", "dep:rand_xoshiro", "nightly"]
# tidy-alphabetical-end
