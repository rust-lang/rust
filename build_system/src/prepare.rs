use std::fs;
use std::path::{Path, PathBuf};

use crate::rustc_info::get_rustc_path;
use crate::utils::{
    cargo_install, create_dir, get_sysroot_dir, git_clone_root_dir, remove_file, run_command,
    run_command_with_output, walk_dir,
};

fn prepare_libcore(
    sysroot_path: &Path,
    libgccjit12_patches: bool,
    cross_compile: bool,
    sysroot_source: Option<String>,
) -> Result<(), String> {
    let rustlib_dir: PathBuf;

    if let Some(path) = sysroot_source {
        rustlib_dir = Path::new(&path)
            .canonicalize()
            .map_err(|error| format!("Failed to canonicalize path: {error:?}"))?;
        if !rustlib_dir.is_dir() {
            return Err(format!("Custom sysroot path {rustlib_dir:?} not found"));
        }
    } else {
        let rustc_path = match get_rustc_path() {
            Some(path) => path,
            None => return Err("`rustc` path not found".to_string()),
        };

        let parent = match rustc_path.parent() {
            Some(path) => path,
            None => return Err(format!("No parent for `{}`", rustc_path.display())),
        };

        rustlib_dir = parent
            .join("../lib/rustlib/src/rust")
            .canonicalize()
            .map_err(|error| format!("Failed to canonicalize path: {error:?}"))?;
        if !rustlib_dir.is_dir() {
            return Err("Please install `rust-src` component".to_string());
        }
    }

    let sysroot_dir = sysroot_path.join("sysroot_src");
    if sysroot_dir.is_dir()
        && let Err(error) = fs::remove_dir_all(&sysroot_dir)
    {
        return Err(format!("Failed to remove `{}`: {:?}", sysroot_dir.display(), error,));
    }

    let sysroot_library_dir = sysroot_dir.join("library");
    create_dir(&sysroot_library_dir)?;

    run_command(&[&"cp", &"-r", &rustlib_dir.join("library"), &sysroot_dir], None)?;

    println!("[GIT] init (cwd): `{}`", sysroot_dir.display());
    run_command(&[&"git", &"init"], Some(&sysroot_dir))?;
    println!("[GIT] add (cwd): `{}`", sysroot_dir.display());
    run_command(&[&"git", &"add", &"."], Some(&sysroot_dir))?;
    println!("[GIT] commit (cwd): `{}`", sysroot_dir.display());

    // This is needed on systems where nothing is configured.
    // git really needs something here, or it will fail.
    // Even using --author is not enough.
    run_command(&[&"git", &"config", &"user.email", &"none@example.com"], Some(&sysroot_dir))?;
    run_command(&[&"git", &"config", &"user.name", &"None"], Some(&sysroot_dir))?;
    run_command(&[&"git", &"config", &"core.autocrlf", &"false"], Some(&sysroot_dir))?;
    run_command(&[&"git", &"config", &"commit.gpgSign", &"false"], Some(&sysroot_dir))?;
    run_command(&[&"git", &"commit", &"-m", &"Initial commit", &"-q"], Some(&sysroot_dir))?;

    let mut patches = Vec::new();
    walk_dir(
        "patches",
        &mut |_| Ok(()),
        &mut |file_path: &Path| {
            patches.push(file_path.to_path_buf());
            Ok(())
        },
        false,
    )?;
    if cross_compile {
        walk_dir(
            "patches/cross_patches",
            &mut |_| Ok(()),
            &mut |file_path: &Path| {
                patches.push(file_path.to_path_buf());
                Ok(())
            },
            false,
        )?;
    }
    if libgccjit12_patches {
        walk_dir(
            "patches/libgccjit12",
            &mut |_| Ok(()),
            &mut |file_path: &Path| {
                patches.push(file_path.to_path_buf());
                Ok(())
            },
            false,
        )?;
    }
    patches.sort();
    for file_path in patches {
        println!("[GIT] apply `{}`", file_path.display());
        let path = Path::new("../../..").join(file_path);
        run_command_with_output(&[&"git", &"apply", &path], Some(&sysroot_dir))?;
        run_command_with_output(&[&"git", &"add", &"-A"], Some(&sysroot_dir))?;
        run_command_with_output(
            &[&"git", &"commit", &"--no-gpg-sign", &"-m", &format!("Patch {}", path.display())],
            Some(&sysroot_dir),
        )?;
    }
    println!("Successfully prepared libcore for building");

    Ok(())
}

// TODO: remove when we can ignore warnings in rustdoc tests.
fn prepare_rand() -> Result<(), String> {
    // Apply patch for the rand crate.
    let file_path = "patches/crates/0001-Remove-deny-warnings.patch";
    let rand_dir = Path::new("build/rand");
    println!("[GIT] apply `{file_path}`");
    let path = Path::new("../..").join(file_path);
    run_command_with_output(&[&"git", &"apply", &path], Some(rand_dir))?;
    run_command_with_output(&[&"git", &"add", &"-A"], Some(rand_dir))?;
    run_command_with_output(
        &[&"git", &"commit", &"--no-gpg-sign", &"-m", &format!("Patch {}", path.display())],
        Some(rand_dir),
    )?;

    Ok(())
}

// build with cg_llvm for perf comparison
fn build_raytracer(repo_dir: &Path) -> Result<(), String> {
    run_command(&[&"cargo", &"build"], Some(repo_dir))?;
    let mv_target = repo_dir.join("raytracer_cg_llvm");
    if mv_target.is_file() {
        remove_file(&mv_target)?;
    }
    run_command(&[&"mv", &"target/debug/main", &"raytracer_cg_llvm"], Some(repo_dir))?;
    Ok(())
}

fn clone_and_setup<F>(repo_url: &str, checkout_commit: &str, extra: Option<F>) -> Result<(), String>
where
    F: Fn(&Path) -> Result<(), String>,
{
    let clone_result = git_clone_root_dir(repo_url, Path::new(crate::BUILD_DIR), false)?;
    if !clone_result.ran_clone {
        println!("`{}` has already been cloned", clone_result.repo_name);
    }
    let repo_path = Path::new(crate::BUILD_DIR).join(&clone_result.repo_name);
    run_command(&[&"git", &"checkout", &"--", &"."], Some(&repo_path))?;
    run_command(&[&"git", &"checkout", &checkout_commit], Some(&repo_path))?;
    if let Some(extra) = extra {
        extra(&repo_path)?;
    }
    Ok(())
}

struct PrepareArg {
    cross_compile: bool,
    only_libcore: bool,
    libgccjit12_patches: bool,
    sysroot_source: Option<String>,
}

impl PrepareArg {
    fn new() -> Result<Option<Self>, String> {
        let mut only_libcore = false;
        let mut cross_compile = false;
        let mut libgccjit12_patches = false;
        let mut sysroot_source = None;

        let mut args = std::env::args().skip(2);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--only-libcore" => only_libcore = true,
                "--cross" => cross_compile = true,
                "--libgccjit12-patches" => libgccjit12_patches = true,
                "--sysroot-source" => {
                    if let Some(path) = args.next() {
                        sysroot_source = Some(path);
                    } else {
                        return Err(
                            "Expected a value after `--sysroot-source`, found nothing".to_string()
                        );
                    }
                }
                "--help" => {
                    Self::usage();
                    return Ok(None);
                }
                a => return Err(format!("Unknown argument `{a}`")),
            }
        }
        Ok(Some(Self { cross_compile, only_libcore, libgccjit12_patches, sysroot_source }))
    }

    fn usage() {
        println!(
            r#"
`prepare` command help:

    --only-libcore           : Only setup libcore and don't clone other repositories
    --cross                  : Apply the patches needed to do cross-compilation
    --libgccjit12-patches    : Apply patches needed for libgccjit12
    --sysroot-source         : Specify custom path for sysroot source
    --help                   : Show this help"#
        )
    }
}

pub fn run() -> Result<(), String> {
    let args = match PrepareArg::new()? {
        Some(a) => a,
        None => return Ok(()),
    };
    let sysroot_path = get_sysroot_dir();
    prepare_libcore(
        &sysroot_path,
        args.libgccjit12_patches,
        args.cross_compile,
        args.sysroot_source,
    )?;

    if !args.only_libcore {
        cargo_install("hyperfine")?;

        let to_clone = &[
            (
                "https://github.com/rust-random/rand.git",
                "1f4507a8e1cf8050e4ceef95eeda8f64645b6719",
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

        prepare_rand()?;
    }

    println!("Successfully ran `prepare`");
    Ok(())
}
