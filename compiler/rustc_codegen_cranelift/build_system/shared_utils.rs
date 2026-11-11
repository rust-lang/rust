// This file is used by both the build system as well as cargo-clif.rs

// Adapted from https://github.com/rust-lang/cargo/blob/6dc1deaddf62c7748c9097c7ea88e9ec77ff1a1a/src/cargo/core/compiler/build_context/target_info.rs#L750-L77
pub(crate) fn rustflags_from_env(kind: &str) -> Vec<String> {
    // First try CARGO_ENCODED_RUSTFLAGS from the environment.
    // Prefer this over RUSTFLAGS since it's less prone to encoding errors.
    if let Ok(a) = std::env::var(format!("CARGO_ENCODED_{}", kind)) {
        if a.is_empty() {
            return Vec::new();
        }
        return a.split('\x1f').map(str::to_string).collect();
    }

    // Then try RUSTFLAGS from the environment
    if let Ok(a) = std::env::var(kind) {
        let args = a.split(' ').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
        return args.collect();
    }

    // No rustflags to be collected from the environment
    Vec::new()
}

pub(crate) fn rustflags_to_cmd_env(cmd: &mut std::process::Command, kind: &str, flags: &[String]) {
    cmd.env(format!("CARGO_ENCODED_{}", kind), flags.join("\x1f"));
}
