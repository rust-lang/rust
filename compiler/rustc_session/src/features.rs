use rustc_target::target_features::Stability;

use crate::Session;

pub trait StabilityExt {
    /// Returns whether the feature may be toggled via `#[target_feature]` or `-Ctarget-feature`.
    /// Otherwise, some features also may only be enabled by flag (target modifier).
    /// (It might still be nightly-only even if this returns `true`, so make sure to also check
    /// `requires_nightly`.)
    fn is_toggle_permitted(&self, sess: &Session) -> Result<(), &'static str>;
}

impl StabilityExt for Stability {
    fn is_toggle_permitted(&self, sess: &Session) -> Result<(), &'static str> {
        match self {
            Stability::Forbidden { reason } => Err(reason),
            Stability::TargetModifierOnly { reason, flag } => {
                if !sess.opts.target_feature_flag_enabled(*flag) { Err(reason) } else { Ok(()) }
            }
            _ => Ok(()),
        }
    }
}
