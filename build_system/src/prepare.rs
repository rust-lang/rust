use crate::rustc_info::get_rustc_path;
use crate::utils::{cargo_install, git_clone, run_command, run_command_with_output, walk_dir};

use std::fs;
use std::path::Path;

fn prepare_libcore(sysroot_path: &Path) -> Result<(), String> {
    let rustc_path = match get_rustc_path() {
        Some(path) => path,
        None => return Err("`rustc` path not found".to_string()),
    };

    let parent = match rustc_path.parent() {
        Some(path) => path,
        None => return Err(format!("No parent for `{}`", rustc_path.display())),
    };

    let rustlib_dir = parent
        .join("../lib/rustlib/src/rust")
        .canonicalize()
        .map_err(|error| format!("Failed to canonicalize path: {:?}", error))?;
    if !rustlib_dir.is_dir() {
        return Err("Please install `rust-src` component".to_string());
    }

    let sysroot_dir = sysroot_path.join("sysroot_src");
    if sysroot_dir.is_dir() {
        if let Err(error) = fs::remove_dir_all(&sysroot_dir) {
            return Err(format!(
                "Failed to remove `{}`: {:?}",
                sysroot_dir.display(),
                error,
            ));
        }
    }

    let sysroot_library_dir = sysroot_dir.join("library");
    fs::create_dir_all(&sysroot_library_dir).map_err(|error| {
        format!(
            "Failed to create folder `{}`: {:?}",
            sysroot_library_dir.display(),
            error,
        )
    })?;

    run_command(
        &[&"cp", &"-r", &rustlib_dir.join("library"), &sysroot_dir],
        None,
    )?;

    println!("[GIT] init (cwd): `{}`", sysroot_dir.display());
    run_command(&[&"git", &"init"], Some(&sysroot_dir))?;
    println!("[GIT] add (cwd): `{}`", sysroot_dir.display());
    run_command(&[&"git", &"add", &"."], Some(&sysroot_dir))?;
    println!("[GIT] commit (cwd): `{}`", sysroot_dir.display());

    // This is needed on systems where nothing is configured.
    // git really needs something here, or it will fail.
    // Even using --author is not enough.
    run_command(
        &[&"git", &"config", &"user.email", &"none@example.com"],
        Some(&sysroot_dir),
    )?;
    run_command(
        &[&"git", &"config", &"user.name", &"None"],
        Some(&sysroot_dir),
    )?;
    run_command(
        &[&"git", &"config", &"core.autocrlf", &"false"],
        Some(&sysroot_dir),
    )?;
    run_command(
        &[&"git", &"config", &"commit.gpgSign", &"false"],
        Some(&sysroot_dir),
    )?;
    run_command(
        &[&"git", &"commit", &"-m", &"Initial commit", &"-q"],
        Some(&sysroot_dir),
    )?;

    let mut patches = Vec::new();
    walk_dir(
        "patches",
        |_| Ok(()),
        |file_path: &Path| {
            patches.push(file_path.to_path_buf());
            Ok(())
        },
    )?;
    patches.sort();
    for file_path in patches {
        println!("[GIT] apply `{}`", file_path.display());
        let path = Path::new("../..").join(file_path);
        run_command_with_output(&[&"git", &"apply", &path], Some(&sysroot_dir))?;
        run_command_with_output(&[&"git", &"add", &"-A"], Some(&sysroot_dir))?;
        run_command_with_output(
            &[
                &"git",
                &"commit",
                &"--no-gpg-sign",
                &"-m",
                &format!("Patch {}", path.display()),
            ],
            Some(&sysroot_dir),
        )?;
    }
    println!("Successfully prepared libcore for building");
    Ok(())
}

// build with cg_llvm for perf comparison
fn build_raytracer(repo_dir: &Path) -> Result<(), String> {
    run_command(&[&"cargo", &"build"], Some(repo_dir))?;
    let mv_target = repo_dir.join("raytracer_cg_llvm");
    if mv_target.is_file() {
        std::fs::remove_file(&mv_target)
            .map_err(|e| format!("Failed to remove file `{}`: {e:?}", mv_target.display()))?;
    }
    run_command(
        &[&"mv", &"target/debug/main", &"raytracer_cg_llvm"],
        Some(repo_dir),
    )?;
    Ok(())
}

fn clone_and_setup<F>(repo_url: &str, checkout_commit: &str, extra: Option<F>) -> Result<(), String>
where
    F: Fn(&Path) -> Result<(), String>,
{
    let clone_result = git_clone(repo_url, None)?;
    if !clone_result.ran_clone {
        println!("`{}` has already been cloned", clone_result.repo_name);
    }
    let repo_path = Path::new(&clone_result.repo_name);
    run_command(&[&"git", &"checkout", &"--", &"."], Some(&repo_path))?;
    run_command(&[&"git", &"checkout", &checkout_commit], Some(&repo_path))?;
    let filter = format!("-{}-", clone_result.repo_name);
    walk_dir(
        "crate_patches",
        |_| Ok(()),
        |file_path| {
            let patch = file_path.as_os_str().to_str().unwrap();
            if patch.contains(&filter) && patch.ends_with(".patch") {
                run_command_with_output(
                    &[&"git", &"am", &file_path.canonicalize().unwrap()],
                    Some(&repo_path),
                )?;
            }
            Ok(())
        },
    )?;
    if let Some(extra) = extra {
        extra(&repo_path)?;
    }
    Ok(())
}

struct PrepareArg {
    only_libcore: bool,
}

impl PrepareArg {
    fn new() -> Result<Option<Self>, String> {
        let mut only_libcore = false;

        for arg in std::env::args().skip(2) {
            match arg.as_str() {
                "--only-libcore" => only_libcore = true,
                "--help" => {
                    Self::usage();
                    return Ok(None);
                }
                a => return Err(format!("Unknown argument `{a}`")),
            }
        }
        Ok(Some(Self { only_libcore }))
    }

    fn usage() {
        println!(
            r#"
`prepare` command help:

    --only-libcore  : Only setup libcore and don't clone other repositories
    --help          : Show this help
"#
        )
    }
}

pub fn run() -> Result<(), String> {
    let args = match PrepareArg::new()? {
        Some(a) => a,
        None => return Ok(()),
    };
    let sysroot_path = Path::new("build_sysroot");
    prepare_libcore(sysroot_path)?;

    if !args.only_libcore {
        cargo_install("hyperfine")?;

        let to_clone = &[
            (
                "https://github.com/rust-random/rand.git",
                "0f933f9c7176e53b2a3c7952ded484e1783f0bf1",
                None,
            ),
            (
                "https://github.com/rust-lang/regex.git",
                "341f207c1071f7290e3f228c710817c280c8dca1",
                None,
            ),
            (
                "https://github.com/ebobby/simple-raytracer",
                "804a7a21b9e673a482797aa289a18ed480e4d813",
                Some(build_raytracer),
            ),
        ];

        for (repo_url, checkout_commit, cb) in to_clone {
            clone_and_setup(repo_url, checkout_commit, *cb)?;
        }
    }

    println!("Successfully ran `prepare`");
    Ok(())
}
