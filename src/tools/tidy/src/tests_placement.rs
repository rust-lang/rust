use std::path::Path;

use crate::diagnostics::DiagCtx;

const FORBIDDEN_PATH: &str = "src/test";
const ALLOWED_PATH: &str = "tests";

pub fn check(root_path: &Path, diag_ctx: DiagCtx) {
    let mut check = diag_ctx.start_check("tests_placement");

    if root_path.join(FORBIDDEN_PATH).exists() {
        check.error(format!(
            "Tests have been moved, please move them from {} to {}",
            root_path.join(FORBIDDEN_PATH).display(),
            root_path.join(ALLOWED_PATH).display()
        ));
    }
}
