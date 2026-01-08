use itertools::Itertools;
use rayon::prelude::*;
use std::{collections::HashMap, process::Command};

pub const INTRINSIC_DELIMITER: &str = "############";
fn runner_command(runner: &str) -> Command {
    let mut it = runner.split_whitespace();
    let mut cmd = Command::new(it.next().unwrap());
    cmd.args(it);

    cmd
}

pub fn compare_outputs(
    intrinsic_name_list: &Vec<String>,
    runner: &str,
    target: &str,
    profile: &str,
) -> bool {
    let profile_dir = match profile {
        "dev" => "debug",
        _ => "release",
    };

    let (c, rust) = rayon::join(
        || {
            runner_command(runner)
                .arg("./intrinsic-test-programs")
                .current_dir("c_programs")
                .output()
        },
        || {
            runner_command(runner)
                .arg(format!(
                    "./target/{target}/{profile_dir}/intrinsic-test-programs"
                ))
                .current_dir("rust_programs")
                .output()
        },
    );
    let (c, rust) = match (c, rust) {
        (Ok(c), Ok(rust)) => (c, rust),
        failure => panic!("Failed to run: {failure:#?}"),
    };

    if !c.status.success() {
        error!(
            "Failed to run C program.\nstdout: {stdout}\nstderr: {stderr}",
            stdout = std::str::from_utf8(&c.stdout).unwrap_or(""),
            stderr = std::str::from_utf8(&c.stderr).unwrap_or(""),
        );
    }

    if !rust.status.success() {
        error!(
            "Failed to run Rust program.\nstdout: {stdout}\nstderr: {stderr}",
            stdout = std::str::from_utf8(&rust.stdout).unwrap_or(""),
            stderr = std::str::from_utf8(&rust.stderr).unwrap_or(""),
        );
    }

    info!("Completed running C++ and Rust test binaries");
    let c = std::str::from_utf8(&c.stdout)
        .unwrap()
        .to_lowercase()
        .replace("-nan", "nan");
    let rust = std::str::from_utf8(&rust.stdout)
        .unwrap()
        .to_lowercase()
        .replace("-nan", "nan");

    let c_output_map = c
        .split(INTRINSIC_DELIMITER)
        .filter_map(|output| output.trim().split_once("\n"))
        .collect::<HashMap<&str, &str>>();
    let rust_output_map = rust
        .split(INTRINSIC_DELIMITER)
        .filter_map(|output| output.trim().split_once("\n"))
        .collect::<HashMap<&str, &str>>();

    let intrinsics = c_output_map
        .keys()
        .chain(rust_output_map.keys())
        .unique()
        .collect_vec();

    info!("Comparing outputs");
    let intrinsics_diff_count = intrinsics
        .par_iter()
        .filter_map(|&&intrinsic| {
            let c_output = c_output_map.get(intrinsic).unwrap();
            let rust_output = rust_output_map.get(intrinsic).unwrap();
            if rust_output.eq(c_output) {
                None
            } else {
                let diff = diff::lines(c_output, rust_output);
                let diffs = diff
                    .into_iter()
                    .filter_map(|diff| match diff {
                        diff::Result::Left(_) | diff::Result::Right(_) => Some(diff),
                        diff::Result::Both(_, _) => None,
                    })
                    .collect_vec();
                if diffs.len() > 0 {
                    Some((intrinsic, diffs))
                } else {
                    None
                }
            }
        })
        .inspect(|(intrinsic, diffs)| {
            println!("Difference for intrinsic: {intrinsic}");
            diffs.into_iter().for_each(|diff| match diff {
                diff::Result::Left(c) => println!("C: {c}"),
                diff::Result::Right(rust) => println!("Rust: {rust}"),
                _ => (),
            });
            println!("****************************************************************");
        })
        .count();

    println!(
        "{} differences found (tested {} intrinsics)",
        intrinsics_diff_count,
        intrinsic_name_list.len()
    );

    intrinsics_diff_count == 0
}
