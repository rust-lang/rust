//! See docs for `FeatureFlags`.

use rustc_hash::FxHashMap;

// FIXME: looks like a much better design is to pass options to each call,
// rather than to have a global ambient feature flags -- that way, the clients
// can issue two successive calls with different options.

/// Feature flags hold fine-grained toggles for all *user-visible* features of
/// rust-analyzer.
///
/// The exists such that users are able to disable any annoying feature (and,
/// with many users and many features,  some features are bound to be annoying
/// for some users)
///
/// Note that we purposefully use run-time checked strings, and not something
/// checked at compile time, to keep things simple and flexible.
///
/// Also note that, at the moment, `FeatureFlags` also store features for
/// `rust-analyzer`. This should be benign layering violation.
#[derive(Debug)]
pub struct FeatureFlags {
    flags: FxHashMap<String, bool>,
}

impl FeatureFlags {
    fn new(flags: &[(&str, bool)]) -> FeatureFlags {
        let flags = flags
            .iter()
            .map(|&(name, value)| {
                check_flag_name(name);
                (name.to_string(), value)
            })
            .collect();
        FeatureFlags { flags }
    }

    pub fn set(&mut self, flag: &str, value: bool) -> Result<(), ()> {
        match self.flags.get_mut(flag) {
            None => Err(()),
            Some(slot) => {
                *slot = value;
                Ok(())
            }
        }
    }

    pub fn get(&self, flag: &str) -> bool {
        match self.flags.get(flag) {
            None => panic!("unknown flag: {:?}", flag),
            Some(value) => *value,
        }
    }
}

impl Default for FeatureFlags {
    fn default() -> FeatureFlags {
        FeatureFlags::new(&[
            ("lsp.diagnostics", true),
            ("completion.insertion.add-call-parenthesis", true),
            ("completion.insertion.add-argument-snippets", true),
            ("completion.enable-postfix", true),
            ("call-info.full", true),
            ("notifications.workspace-loaded", true),
            ("notifications.cargo-toml-not-found", true),
        ])
    }
}

fn check_flag_name(flag: &str) {
    for c in flag.bytes() {
        match c {
            b'a'..=b'z' | b'-' | b'.' => (),
            _ => panic!("flag name does not match conventions: {:?}", flag),
        }
    }
}
