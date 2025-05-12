use std::path::Path;

const FORBIDDEN_PATH: &str = "src/test";
const ALLOWED_PATH: &str = "tests";

pub fn check(root_path: impl AsRef<Path>, bad: &mut bool) {
    if root_path.as_ref().join(FORBIDDEN_PATH).exists() {
        tidy_error!(
            bad,
            "Tests have been moved, please move them from {} to {}",
            root_path.as_ref().join(FORBIDDEN_PATH).display(),
            root_path.as_ref().join(ALLOWED_PATH).display()
        )
    }
}
