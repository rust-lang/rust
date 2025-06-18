use rustc_target::target_features::Stability;

use crate::Session;
use crate::errors::{ForbiddenCTargetFeature, UnstableCTargetFeature};

pub trait StabilityExt {
    /// Returns whether the feature may be toggled via `#[target_feature]` or `-Ctarget-feature`.
    /// Otherwise, some features also may only be enabled by flag (target modifier).
    /// (It might still be nightly-only even if this returns `true`, so make sure to also check
    /// `requires_nightly`.)
    fn is_toggle_permitted(&self, sess: &Session) -> Result<(), &'static str>;

    /// Check that feature is correctly enabled/disabled by command line flag (emits warnings)
    fn verify_feature_enabled_by_flag(&self, sess: &Session, enable: bool, feature: &str);
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
    fn verify_feature_enabled_by_flag(&self, sess: &Session, enable: bool, feature: &str) {
        if let Err(reason) = self.is_toggle_permitted(sess) {
            sess.dcx().emit_warn(ForbiddenCTargetFeature {
                feature,
                enabled: if enable { "enabled" } else { "disabled" },
                reason,
            });
        } else if self.requires_nightly().is_some() {
            // An unstable feature. Warn about using it. It makes little sense
            // to hard-error here since we just warn about fully unknown
            // features above.
            sess.dcx().emit_warn(UnstableCTargetFeature { feature });
        }
    }
}

pub fn retpoline_features_by_flags(sess: &Session, features: &mut Vec<&str>) {
    // -Zretpoline without -Zretpoline-external-thunk enables
    // retpoline-indirect-branches and retpoline-indirect-calls target features
    let unstable_opts = &sess.opts.unstable_opts;
    if unstable_opts.retpoline && !unstable_opts.retpoline_external_thunk {
        features.push("+retpoline-indirect-branches");
        features.push("+retpoline-indirect-calls");
    }
    // -Zretpoline-external-thunk (maybe, with -Zretpoline too) enables
    // retpoline-external-thunk, retpoline-indirect-branches and
    // retpoline-indirect-calls target features
    if unstable_opts.retpoline_external_thunk {
        features.push("+retpoline-external-thunk");
        features.push("+retpoline-indirect-branches");
        features.push("+retpoline-indirect-calls");
    }
}
