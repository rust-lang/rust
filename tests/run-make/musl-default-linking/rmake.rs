use run_make_support::{rustc, serde_json};

// Please do NOT add more targets to this list!
// Per https://github.com/rust-lang/compiler-team/issues/422,
// we should be trying to move these targets to dynamically link
// musl libc by default.
//@ needs-llvm-components: aarch64 arm powerpc x86
static LEGACY_STATIC_LINKING_TARGETS: &[&'static str] = &[
    "aarch64-unknown-linux-musl",
    "arm-unknown-linux-musleabi",
    "arm-unknown-linux-musleabihf",
    "armv5te-unknown-linux-musleabi",
    "armv7-unknown-linux-musleabi",
    "armv7-unknown-linux-musleabihf",
    "i586-unknown-linux-musl",
    "i686-unknown-linux-musl",
    "powerpc64le-unknown-linux-musl",
    "x86_64-unknown-linux-musl",
];

fn main() {
    let targets = rustc().print("target-list").run().stdout_utf8();

    for target in targets.lines() {
        let abi = target.split('-').last().unwrap();

        if !abi.starts_with("musl") {
            continue;
        }

        let target_spec_json = rustc()
            .print("target-spec-json")
            .target(target)
            .arg("-Zunstable-options")
            .run()
            .stdout_utf8();

        let target_spec: serde_json::Value =
            serde_json::from_str(&target_spec_json).expect("failed to parse target-spec-json");

        let target_families = &target_spec["target-family"];
        // WebAssembly doesn't support dynamic linking yet; all musl targets
        // need to be statically linked.
        if target_families
            .as_array()
            .expect("target-family wasn't an array")
            .iter()
            .filter_map(|x| x.as_str())
            .any(|family| family == "wasm")
        {
            continue;
        }

        let default = &target_spec["crt-static-default"];
        // If the value is `null`, then the default to dynamically link from
        // musl_base was not overridden.
        if default.is_null() {
            continue;
        }

        if default.as_bool().expect("wasn't a boolean")
            && !LEGACY_STATIC_LINKING_TARGETS.contains(&target)
        {
            panic!("{target} statically links musl libc when it should dynamically link it");
        }
    }
}
