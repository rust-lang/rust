// feature removed error
#![feature(old_bad_feature1)] //~ ERROR the feature `old_bad_feature1` has been removed
#![unstable_removed(
    feature = "old_bad_feature1",
    since = "1.80.0",
    issue = "12345",
    reason = "it was replaced by the 'new_good_feature' gate"
)]

// unknown feature as there's no unstable_removed relating this
#![feature(old_bad_feature2)] //~ ERROR[E0635]

// no error as this feature is never enabled using #![feature()]
#![unstable_removed(
    feature = "old_bad_feature3",
    since = "1.80.0",
    issue = "12345",
    reason = "it was replaced by the 'new_good_feature' gate"
)]

fn main() {}
