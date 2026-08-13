use crate::common::run_command_with_env;

use std::collections::HashMap;

fn run_tests_in_dir(env: &HashMap<&str, &str>, dir: &str) -> Result<(), String> {
    run_command_with_env("cargo", &["build", "--locked"], dir, &env)?;
    run_command_with_env("cargo", &["test"], dir, &env)
}

pub fn runner() -> Result<(), String> {
    let Ok(rustflags) = std::env::var("RUSTFLAGS") else {
        return Err(
            "`RUSTFLAGS` environment variable must be set to run `build-and-test`".to_string(),
        );
    };
    if !rustflags.contains("-D warnings") && !rustflags.contains("-Dwarnings") {
        return Err(
            "`RUSTFLAGS` environment variable must contain `-Dwarnings` to run `build-and-test`"
                .to_string(),
        );
    }

    let mut env = HashMap::from([("RUSTFLAGS", "-D warnings"), ("RUSTFMT_CI", "1")]);
    let value_holder;
    if let Ok(cfg_release_channel) = std::env::var("CFG_RELEASE_CHANNEL") {
        value_holder = cfg_release_channel;
        env.insert("CFG_RELEASE_CHANNEL", value_holder.as_str());
    }

    // Print version information
    run_command_with_env("rustc", &["-Vv"], ".", &env)?;
    run_command_with_env("cargo", &["-v"], ".", &env)?;

    // Build and test main crate
    let options: &[&str] =
        if std::env::var("CFG_RELEASE_CHANNEL").is_ok_and(|value| value == "nightly") {
            &["build", "--locked", "--all-features"]
        } else {
            &["build", "--locked"]
        };
    run_command_with_env("cargo", options, ".", &env)?;
    run_command_with_env("cargo", &["test"], ".", &env)?;

    // Build and test config_proc_macro
    run_tests_in_dir(&env, "config_proc_macro")?;
    run_tests_in_dir(&env, "check_diff")?;

    Ok(())
}
