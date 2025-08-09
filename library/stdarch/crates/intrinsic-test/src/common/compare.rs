use super::cli::FailureReason;
use rayon::prelude::*;
use std::process::Command;

fn runner_command(runner: &str) -> Command {
    let mut it = runner.split_whitespace();
    let mut cmd = Command::new(it.next().unwrap());
    cmd.args(it);

    cmd
}

pub fn compare_outputs(intrinsic_name_list: &Vec<String>, runner: &str, target: &str) -> bool {
    let intrinsics = intrinsic_name_list
        .par_iter()
        .filter_map(|intrinsic_name| {

            let c = runner_command(runner)
                .arg("intrinsic-test-programs")
                .arg(intrinsic_name)
                .current_dir("c_programs")
                .output();

            let rust = runner_command(runner)
                .arg(format!("target/{target}/release/intrinsic-test-programs"))
                .arg(intrinsic_name)
                .current_dir("rust_programs")
                .output();

            let (c, rust) = match (c, rust) {
                (Ok(c), Ok(rust)) => (c, rust),
                a => panic!("{a:#?}"),
            };

            if !c.status.success() {
                error!(
                    "Failed to run C program for intrinsic `{intrinsic_name}`\nstdout: {stdout}\nstderr: {stderr}",
                    stdout = std::str::from_utf8(&c.stdout).unwrap_or(""),
                    stderr = std::str::from_utf8(&c.stderr).unwrap_or(""),
                );
                return Some(FailureReason::RunC(intrinsic_name.clone()));
            }

            if !rust.status.success() {
                error!(
                    "Failed to run Rust program for intrinsic `{intrinsic_name}`\nstdout: {stdout}\nstderr: {stderr}",
                    stdout = std::str::from_utf8(&rust.stdout).unwrap_or(""),
                    stderr = std::str::from_utf8(&rust.stderr).unwrap_or(""),
                );
                return Some(FailureReason::RunRust(intrinsic_name.clone()));
            }

            info!("Comparing intrinsic: {intrinsic_name}");

            let c = std::str::from_utf8(&c.stdout)
                .unwrap()
                .to_lowercase()
                .replace("-nan", "nan");
            let rust = std::str::from_utf8(&rust.stdout)
                .unwrap()
                .to_lowercase()
                .replace("-nan", "nan");

            if c == rust {
                None
            } else {
                Some(FailureReason::Difference(intrinsic_name.clone(), c, rust))
            }
        })
        .collect::<Vec<_>>();

    intrinsics.iter().for_each(|reason| match reason {
        FailureReason::Difference(intrinsic, c, rust) => {
            println!("Difference for intrinsic: {intrinsic}");
            let diff = diff::lines(c, rust);
            diff.iter().for_each(|diff| match diff {
                diff::Result::Left(c) => println!("C: {c}"),
                diff::Result::Right(rust) => println!("Rust: {rust}"),
                diff::Result::Both(_, _) => (),
            });
            println!("****************************************************************");
        }
        FailureReason::RunC(intrinsic) => {
            println!("Failed to run C program for intrinsic {intrinsic}")
        }
        FailureReason::RunRust(intrinsic) => {
            println!("Failed to run rust program for intrinsic {intrinsic}")
        }
    });
    println!("{} differences found", intrinsics.len());
    intrinsics.is_empty()
}
