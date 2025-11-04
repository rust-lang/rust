use super::cli::FailureReason;
use itertools::Itertools;
use rayon::prelude::*;
use std::{collections::HashMap, process::Command};

fn runner_command(runner: &str) -> Command {
    let mut it = runner.split_whitespace();
    let mut cmd = Command::new(it.next().unwrap());
    cmd.args(it);

    cmd
}

pub fn compare_outputs(intrinsic_name_list: &Vec<String>, runner: &str, target: &str) -> bool {
    let c = runner_command(runner)
        .arg("./intrinsic-test-programs")
        .current_dir("c_programs")
        .output();

    let rust = runner_command(runner)
        .arg(format!("./target/{target}/release/intrinsic-test-programs"))
        .current_dir("rust_programs")
        .output();
    
    let (c, rust) = match (c, rust) {
        (Ok(c), Ok(rust)) => (c, rust),
        a => panic!("{a:#?}"),
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

    let c = std::str::from_utf8(&c.stdout)
        .unwrap()
        .to_lowercase()
        .replace("-nan", "nan");
    let rust = std::str::from_utf8(&rust.stdout)
        .unwrap()
        .to_lowercase()
        .replace("-nan", "nan");
    
    let c_output_map = c.split("############")
        .filter_map(|output| output.trim().split_once("\n"))
        .collect::<HashMap<&str, &str>>();
    let rust_output_map = rust.split("############")
        .filter_map(|output| output.trim().split_once("\n"))
        .collect::<HashMap<&str, &str>>();

    let intrinsics = c_output_map.keys().chain(rust_output_map.keys()).unique().collect_vec();
    let intrinsics_diff_count = intrinsics
        .par_iter()
        .filter_map(|&&intrinsic| {
            println!("Difference for intrinsic: {intrinsic}");
            let c_output = c_output_map.get(intrinsic).unwrap();
            let rust_output = rust_output_map.get(intrinsic).unwrap();
            let diff = diff::lines(c_output, rust_output);
            let diff_count = diff.into_iter().filter_map(|diff| match diff {
                diff::Result::Left(c) => {
                    println!("C: {c}");
                    Some(c)
                }
                diff::Result::Right(rust) => {
                    println!("Rust: {rust}");
                    Some(rust)
                }
                diff::Result::Both(_, _) => None,
            }).count();
            println!("****************************************************************");
            if diff_count > 0 {
                Some(intrinsic)
            } else { None }
        }).count();

    println!(
        "{} differences found (tested {} intrinsics)",
        intrinsics_diff_count,
        intrinsic_name_list.len()
    );

    intrinsics_diff_count == 0
}
