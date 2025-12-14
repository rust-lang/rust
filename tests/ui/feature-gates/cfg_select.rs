#![warn(unreachable_cfgs)] // Unused warnings are disabled by default in UI tests.

// `#[feature(cfg_select)]` is a libs feature (so, not a lang feature), but it lints on unreachable
// branches, and that lint should only be emitted when the feature is enabled.

cfg_select! {
    //~^ ERROR use of unstable library feature `cfg_select`
    _ => {}
    true => {} // With the feature enabled, this branch would trip the unreachable_cfgs lint.
}

fn main() {}
