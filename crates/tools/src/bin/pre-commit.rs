use std::process::Command;

use failure::bail;

use tools::{Result, run_rustfmt, run, project_root};

fn main() -> tools::Result<()> {
    run_rustfmt(tools::Overwrite)?;
    update_staged()
}

fn update_staged() -> Result<()> {
    let root = project_root();
    let output = Command::new("git")
        .arg("diff")
        .arg("--diff-filter=MAR")
        .arg("--name-only")
        .arg("--cached")
        .current_dir(&root)
        .output()?;
    if !output.status.success() {
        bail!(
            "`git diff --diff-filter=MAR --name-only --cached` exited with {}",
            output.status
        );
    }
    for line in String::from_utf8(output.stdout)?.lines() {
        run(
            &format!(
                "git update-index --add {}",
                root.join(line).to_string_lossy()
            ),
            ".",
        )?;
    }
    Ok(())
}
