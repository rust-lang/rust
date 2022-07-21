use color_eyre::eyre::{bail, Result};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    process::Command,
};

use crate::Config;

#[derive(Default, Debug)]
pub struct Dependencies {
    /// All paths that must be imported with `-L dependency=`. This is for
    /// finding proc macros run on the host and dependencies for the target.
    pub import_paths: Vec<PathBuf>,
    /// The name as chosen in the `Cargo.toml` and its corresponding rmeta file.
    pub dependencies: Vec<(String, PathBuf)>,
}

/// Compiles dependencies and returns the crate names and corresponding rmeta files.
pub fn build_dependencies(config: &Config) -> Result<Dependencies> {
    let manifest_path = match &config.dependencies_crate_manifest_path {
        Some(path) => path,
        None => return Ok(Default::default()),
    };
    let (program, args, envs): (&Path, &[_], &[_]) = match &config.dependency_builder {
        Some(db) => (&db.program, &db.args, &db.envs),
        None => (Path::new("cargo"), &[], &[]),
    };
    let mut build = Command::new(program);
    build.args(args);
    // HACK: we're using `cargo run` (or `cargo miri run`), because the latter does not
    // support `cargo miri build` yet.
    build.arg("run");

    if let Some(target) = &config.target {
        build.arg(format!("--target={target}"));
    }

    // Reusable closure for setting up the environment both for artifact generation and `cargo_metadata`
    let setup_command = |cmd: &mut Command| {
        cmd.envs(envs.iter().map(|(k, v)| (k, v)));
        cmd.arg("--manifest-path").arg(manifest_path);
    };

    setup_command(&mut build);
    build
        .arg("--target-dir=target/test_dependencies")
        .arg("--message-format=json");

    let output = build.output()?;

    if !output.status.success() {
        let stdout = String::from_utf8(output.stdout)?;
        let stderr = String::from_utf8(output.stderr)?;
        bail!("failed to compile dependencies:\nstderr:\n{stderr}\n\nstdout:{stdout}");
    }

    // Collect all artifacts generated
    let output = output.stdout;
    let output = String::from_utf8(output)?;
    let mut import_paths: HashSet<PathBuf> = HashSet::new();
    let mut artifacts: HashMap<_, _> = output
        .lines()
        .filter_map(|line| {
            let message = serde_json::from_str::<cargo_metadata::Message>(line).ok()?;
            if let cargo_metadata::Message::CompilerArtifact(artifact) = message {
                for filename in &artifact.filenames {
                    import_paths.insert(filename.parent().unwrap().into());
                }
                let filename = artifact
                    .filenames
                    .into_iter()
                    .find(|filename| filename.extension() == Some("rmeta"))?;
                Some((artifact.package_id, filename.into_std_path_buf()))
            } else {
                None
            }
        })
        .collect();

    // Check which crates are mentioned in the crate itself
    let mut metadata = cargo_metadata::MetadataCommand::new().cargo_command();
    setup_command(&mut metadata);
    let output = metadata.output()?;

    if !output.status.success() {
        let stdout = String::from_utf8(output.stdout)?;
        let stderr = String::from_utf8(output.stderr)?;
        bail!("failed to run cargo-metadata:\nstderr:\n{stderr}\n\nstdout:{stdout}");
    }

    let output = output.stdout;
    let output = String::from_utf8(output)?;

    for line in output.lines() {
        if !line.starts_with('{') {
            continue;
        }
        let metadata: cargo_metadata::Metadata = serde_json::from_str(line)?;
        // Only take artifacts that are defined in the Cargo.toml

        // First, find the root artifact
        let root = metadata
            .packages
            .iter()
            .find(|package| {
                package.manifest_path.as_std_path().canonicalize().unwrap()
                    == manifest_path.canonicalize().unwrap()
            })
            .unwrap();

        // Then go over all of its dependencies
        let dependencies = root
            .dependencies
            .iter()
            .map(|package| {
                // Get the id for the package matching the version requirement of the dep
                let id = &metadata
                    .packages
                    .iter()
                    .find(|&dep| dep.name == package.name && package.req.matches(&dep.version))
                    .expect("dependency does not exist")
                    .id;
                // Return the name chosen in `Cargo.toml` and the path to the corresponding artifact
                (
                    package.rename.clone().unwrap_or_else(|| package.name.clone()),
                    artifacts.remove(id).expect("package without artifact"),
                )
            })
            .collect();
        let import_paths = import_paths.into_iter().collect();
        return Ok(Dependencies { dependencies, import_paths });
    }

    bail!("no json found in cargo-metadata output")
}
