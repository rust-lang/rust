#![deny(clippy::empty_loop)]

#[cfg(feature = "primary_package_test")]
pub fn lint_me() {
    loop {}
}
