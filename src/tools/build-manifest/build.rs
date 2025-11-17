use std::fmt::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Default, Debug)]
pub(crate) struct RustcTargets {
    /// Targets with host tool artifacts.
    pub(crate) hosts: Vec<String>,

    /// All targets we distribute some artifacts for (superset of `hosts`).
    pub(crate) targets: Vec<String>,
}

fn collect_rustc_targets() -> RustcTargets {
    let rustc_path = std::env::var("RUSTC").expect("RUSTC set");
    let output = Command::new(&rustc_path)
        .arg("--print=all-target-specs-json")
        .env("RUSTC_BOOTSTRAP", "1")
        .arg("-Zunstable-options")
        .stderr(Stdio::inherit())
        .output()
        .unwrap();
    assert!(output.status.success());
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();

    let mut rustc_targets = RustcTargets::default();
    for (target, json) in json.as_object().unwrap().iter() {
        let Some(tier) = json["metadata"]["tier"].as_u64() else {
            eprintln!("skipping {target}: no tier in metadata");
            continue;
        };
        let host_tools: Option<bool> =
            serde_json::from_value(json["metadata"]["host_tools"].clone()).unwrap();

        if !(tier == 1 || tier == 2) {
            eprintln!("ignoring {target}: tier {tier} insufficient for target to be in manifest");
            continue;
        }

        if host_tools == Some(true) {
            rustc_targets.hosts.push(target.to_owned());
            rustc_targets.targets.push(target.to_owned());
        } else {
            rustc_targets.targets.push(target.to_owned());
        }
    }

    rustc_targets
}

fn main() {
    let targets = collect_rustc_targets();

    // Verify we ended up with a reasonable target list.
    assert!(targets.hosts.len() >= 10);
    assert!(targets.targets.len() >= 30);
    assert!(targets.hosts.iter().any(|e| e == "x86_64-unknown-linux-gnu"));
    assert!(targets.targets.iter().any(|e| e == "x86_64-unknown-linux-gnu"));

    let mut output = String::new();

    writeln!(output, "static HOSTS: &[&str] = &[").unwrap();
    for host in targets.hosts {
        writeln!(output, "    {:?},", host).unwrap();
    }
    writeln!(output, "];").unwrap();

    writeln!(output, "static TARGETS: &[&str] = &[").unwrap();
    for target in targets.targets {
        writeln!(output, "    {:?},", target).unwrap();
    }
    writeln!(output, "];").unwrap();

    std::fs::write(PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("targets.rs"), output)
        .unwrap();
}
