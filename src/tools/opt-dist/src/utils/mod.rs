use std::time::Duration;

use humansize::BINARY;
use sysinfo::Disks;

use crate::environment::Environment;
use crate::timer::Timer;
use crate::utils::io::delete_directory;

pub mod artifact_size;
pub mod io;

pub fn format_env_variables() -> String {
    let vars = std::env::vars().map(|(key, value)| format!("{key}={value}")).collect::<Vec<_>>();
    vars.join("\n")
}

pub fn print_free_disk_space() -> anyhow::Result<()> {
    let disks = Disks::new_with_refreshed_list();
    let available_space: u64 = disks.list().iter().map(|d| d.available_space()).sum();
    let total_space: u64 = disks.list().iter().map(|d| d.total_space()).sum();
    let used_space = total_space - available_space;

    log::info!(
        "Free disk space: {} out of total {} ({:.2}% used)",
        humansize::format_size(available_space, BINARY),
        humansize::format_size(total_space, BINARY),
        (used_space as f64 / total_space as f64) * 100.0
    );
    Ok(())
}

pub fn clear_llvm_files(env: &Environment) -> anyhow::Result<()> {
    // Bootstrap currently doesn't support rebuilding LLVM when PGO options
    // change (or any other llvm-related options); so just clear out the relevant
    // directories ourselves.
    log::info!("Clearing LLVM build files");
    delete_directory(&env.build_artifacts().join("llvm"))?;
    if env.build_artifacts().join("lld").is_dir() {
        delete_directory(&env.build_artifacts().join("lld"))?;
    }
    Ok(())
}

/// Write the formatted statistics of the timer to a Github Actions summary.
pub fn write_timer_to_summary(path: &str, timer: &Timer) -> anyhow::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::options().append(true).create(true).open(path)?;
    writeln!(
        file,
        r#"# Step durations

```
{}
```
"#,
        timer.format_stats()
    )?;
    Ok(())
}

/// Wraps all output produced within the `func` closure in a CI output group, if we're running in
/// CI.
pub fn with_log_group<F: FnOnce() -> R, R>(group: &str, func: F) -> R {
    if is_in_ci() {
        println!("::group::{group}");
        let result = func();
        println!("::endgroup::");
        result
    } else {
        func()
    }
}

#[allow(unused)]
pub fn retry_action<F: Fn() -> anyhow::Result<R>, R>(
    action: F,
    name: &str,
    count: u64,
) -> anyhow::Result<R> {
    for attempt in 0..count {
        match action() {
            Ok(result) => return Ok(result),
            Err(error) => {
                log::error!("Failed to perform action `{name}`, attempt #{attempt}: {error:?}");
                std::thread::sleep(Duration::from_secs(5));
            }
        }
    }
    Err(anyhow::anyhow!("Failed to perform action `{name}` after {count} retries"))
}

fn is_in_ci() -> bool {
    std::env::var("GITHUB_ACTIONS").is_ok()
}
