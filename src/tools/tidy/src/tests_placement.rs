use std::path::Path;

use crate::diagnostics::TidyCtx;

const FORBIDDEN_PATH: &str = "src/test";
const ALLOWED_PATH: &str = "tests";

pub fn check(root_path: &Path, tidy_ctx: TidyCtx) {
    let mut check = tidy_ctx.start_check("tests_placement");

    if root_path.join(FORBIDDEN_PATH).exists() {
        check.error(format!(
            "Tests have been moved, please move them from {} to {}",
            root_path.join(FORBIDDEN_PATH).display(),
            root_path.join(ALLOWED_PATH).display()
        ));
    }
}
