#![stable(feature = "stable_feature", since = "1.0.0")]
#![feature(staged_api)]
#![crate_type = "lib"]

#[unstable(feature = "a", issue = "1", soft)]
#[unstable(feature = "b", issue = "2", reason = "reason", soft)]
#[macro_export]
macro_rules! mac {
    () => ()
}

#[unstable(feature = "c", issue = "3", soft)]
#[unstable(feature = "d", issue = "4", reason = "reason", soft)]
pub fn something() {}
