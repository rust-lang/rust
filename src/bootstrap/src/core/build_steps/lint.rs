use crate::core::builder::Builder;
use std::path::PathBuf;

pub fn lint(build: &Builder<'_>, paths: &[PathBuf]) {
    if build.config.dry_run() {
        return;
    }

    let compiler = build.compiler(0, build.config.build);

    for path in paths {
        let mut clippy = build.cargo_clippy_cmd(compiler);
        clippy.current_dir(path);

        clippy.env("RUSTC_BOOTSTRAP", "1");
        clippy.args(["--all-targets", "--all-features", "--", "--D", "warnings"]);
        build.run(&mut clippy);
    }
}
