use std::{fs, path::Path, process::Command};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// This tool builds the github-pages website to the `./target/website` folder
fn main() {
    if let Err(err) = try_main() {
        eprintln!("{}", err);
        std::process::exit(-1);
    }
}

fn try_main() -> Result<()> {
    check_cwd()?;
    build_scaffold()?;
    build_docs()?;
    println!("Finished\n./target/website/index.html");
    Ok(())
}

fn cargo() -> Command {
    Command::new("cargo")
}

fn check_cwd() -> Result<()> {
    let toml = std::fs::read_to_string("./Cargo.toml")?;
    if !toml.contains("[workspace]") {
        Err("website-gen should be run from the root of workspace")?;
    }
    Ok(())
}

fn build_docs() -> Result<()> {
    let status = cargo().args(&["doc", "--all", "--no-deps"]).status()?;
    if !status.success() {
        Err("cargo doc failed")?;
    }
    sync_dir("./target/doc", "./target/website/api-docs")?;
    Ok(())
}

fn build_scaffold() -> Result<()> {
    sync_dir("./website/src", "./target/website")
}

fn sync_dir(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<()> {
    return sync_dir(src.as_ref(), dst.as_ref());

    fn sync_dir(src: &Path, dst: &Path) -> Result<()> {
        let _ = fs::remove_dir_all(dst);
        fs::create_dir_all(dst)?;
        for entry in walkdir::WalkDir::new(src) {
            let entry = entry?;
            let src_path = entry.path();
            let dst_path = dst.join(src_path.strip_prefix(src)?);
            if src_path.is_dir() {
                fs::create_dir_all(dst_path)?;
            } else {
                fs::copy(src_path, dst_path)?;
            }
        }
        Ok(())
    }
}
