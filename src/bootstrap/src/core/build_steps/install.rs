//! Implementation of the install aspects of the compiler.
//!
//! This module is responsible for installing the standard library,
//! compiler, and documentation.

use std::path::{Component, Path, PathBuf};
use std::{env, fs};

use crate::core::build_steps::dist;
use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::core::config::{Config, TargetSelection};
use crate::utils::exec::command;
use crate::utils::helpers::t;
use crate::utils::tarball::GeneratedTarball;
use crate::{Compiler, Kind};

#[cfg(target_os = "illumos")]
const SHELL: &str = "bash";
#[cfg(not(target_os = "illumos"))]
const SHELL: &str = "sh";

/// We have to run a few shell scripts, which choke quite a bit on both `\`
/// characters and on `C:\` paths, so normalize both of them away.
fn sanitize_sh(path: &Path, is_cygwin: bool) -> String {
    let path = path.to_str().unwrap().replace('\\', "/");
    return if is_cygwin { path } else { change_drive(unc_to_lfs(&path)).unwrap_or(path) };

    fn unc_to_lfs(s: &str) -> &str {
        s.strip_prefix("//?/").unwrap_or(s)
    }

    fn change_drive(s: &str) -> Option<String> {
        let mut ch = s.chars();
        let drive = ch.next().unwrap_or('C');
        if ch.next() != Some(':') {
            return None;
        }
        if ch.next() != Some('/') {
            return None;
        }
        // The prefix for Windows drives in Cygwin/MSYS2 is configurable, but
        // /proc/cygdrive is available regardless of configuration since 1.7.33
        Some(format!("/proc/cygdrive/{}/{}", drive, &s[drive.len_utf8() + 2..]))
    }
}

fn is_dir_writable_for_user(dir: &Path) -> bool {
    let tmp = dir.join(".tmp");
    match fs::create_dir_all(&tmp) {
        Ok(_) => {
            fs::remove_dir_all(tmp).unwrap();
            true
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                false
            } else {
                panic!("Failed the write access check for the current user. {e}");
            }
        }
    }
}

fn install_sh(
    builder: &Builder<'_>,
    package: &str,
    stage: u32,
    host: Option<TargetSelection>,
    tarball: &GeneratedTarball,
) {
    let _guard = builder.msg(Kind::Install, stage, package, host, host);

    let prefix = default_path(&builder.config.prefix, "/usr/local");
    let sysconfdir = prefix.join(default_path(&builder.config.sysconfdir, "/etc"));
    let destdir_env = env::var_os("DESTDIR").map(PathBuf::from);
    let is_cygwin = builder.config.build.is_cygwin();

    // Sanity checks on the write access of user.
    //
    // When the `DESTDIR` environment variable is present, there is no point to
    // check write access for `prefix` and `sysconfdir` individually, as they
    // are combined with the path from the `DESTDIR` environment variable. In
    // this case, we only need to check the `DESTDIR` path, disregarding the
    // `prefix` and `sysconfdir` paths.
    if let Some(destdir) = &destdir_env {
        assert!(is_dir_writable_for_user(destdir), "User doesn't have write access on DESTDIR.");
    } else {
        assert!(
            is_dir_writable_for_user(&prefix),
            "User doesn't have write access on `install.prefix` path in the `bootstrap.toml`.",
        );
        assert!(
            is_dir_writable_for_user(&sysconfdir),
            "User doesn't have write access on `install.sysconfdir` path in `bootstrap.toml`."
        );
    }

    let datadir = prefix.join(default_path(&builder.config.datadir, "share"));
    let docdir = prefix.join(default_path(&builder.config.docdir, &format!("share/doc/{package}")));
    let mandir = prefix.join(default_path(&builder.config.mandir, "share/man"));
    let libdir = prefix.join(default_path(&builder.config.libdir, "lib"));
    let bindir = prefix.join(&builder.config.bindir); // Default in config.rs

    let empty_dir = builder.out.join("tmp/empty_dir");
    t!(fs::create_dir_all(&empty_dir));

    let mut cmd = command(SHELL);
    cmd.current_dir(&empty_dir)
        .arg(sanitize_sh(&tarball.decompressed_output().join("install.sh"), is_cygwin))
        .arg(format!("--prefix={}", prepare_dir(&destdir_env, prefix, is_cygwin)))
        .arg(format!("--sysconfdir={}", prepare_dir(&destdir_env, sysconfdir, is_cygwin)))
        .arg(format!("--datadir={}", prepare_dir(&destdir_env, datadir, is_cygwin)))
        .arg(format!("--docdir={}", prepare_dir(&destdir_env, docdir, is_cygwin)))
        .arg(format!("--bindir={}", prepare_dir(&destdir_env, bindir, is_cygwin)))
        .arg(format!("--libdir={}", prepare_dir(&destdir_env, libdir, is_cygwin)))
        .arg(format!("--mandir={}", prepare_dir(&destdir_env, mandir, is_cygwin)))
        .arg("--disable-ldconfig");
    cmd.run(builder);
    t!(fs::remove_dir_all(&empty_dir));
}

fn default_path(config: &Option<PathBuf>, default: &str) -> PathBuf {
    config.as_ref().cloned().unwrap_or_else(|| PathBuf::from(default))
}

fn prepare_dir(destdir_env: &Option<PathBuf>, mut path: PathBuf, is_cygwin: bool) -> String {
    // The DESTDIR environment variable is a standard way to install software in a subdirectory
    // while keeping the original directory structure, even if the prefix or other directories
    // contain absolute paths.
    //
    // More information on the environment variable is available here:
    // https://www.gnu.org/prep/standards/html_node/DESTDIR.html
    if let Some(destdir) = destdir_env {
        let without_destdir = path.clone();
        path.clone_from(destdir);
        // Custom .join() which ignores disk roots.
        for part in without_destdir.components() {
            if let Component::Normal(s) = part {
                path.push(s)
            }
        }
    }

    // The installation command is not executed from the current directory, but from a temporary
    // directory. To prevent relative paths from breaking this converts relative paths to absolute
    // paths. std::fs::canonicalize is not used as that requires the path to actually be present.
    if path.is_relative() {
        path = std::env::current_dir().expect("failed to get the current directory").join(path);
        assert!(path.is_absolute(), "could not make the path relative");
    }

    sanitize_sh(&path, is_cygwin)
}

macro_rules! install {
    (($sel:ident, $builder:ident, $_config:ident),
       $($name:ident,
       $condition_name: ident = $path_or_alias: literal,
       $default_cond:expr,
       only_hosts: $only_hosts:expr,
       $run_item:block $(, $c:ident)*;)+) => {
        $(
            #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            pub compiler: Compiler,
            pub target: TargetSelection,
        }

        impl $name {
            #[allow(dead_code)]
            fn should_build(config: &Config) -> bool {
                config.extended && config.tools.as_ref()
                    .map_or(true, |t| t.contains($path_or_alias))
            }
        }

        impl Step for $name {
            type Output = ();
            const DEFAULT: bool = true;
            const ONLY_HOSTS: bool = $only_hosts;
            $(const $c: bool = true;)*

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                let $_config = &run.builder.config;
                run.$condition_name($path_or_alias).default_condition($default_cond)
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($name {
                    compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
                    target: run.target,
                });
            }

            fn run($sel, $builder: &Builder<'_>) {
                $run_item
            }
        })+
    }
}

install!((self, builder, _config),
    Docs, path = "src/doc", _config.docs, only_hosts: false, {
        let tarball = builder.ensure(dist::Docs { host: self.target }).expect("missing docs");
        install_sh(builder, "docs", self.compiler.stage, Some(self.target), &tarball);
    };
    Std, path = "library/std", true, only_hosts: false, {
        // `expect` should be safe, only None when host != build, but this
        // only runs when host == build
        let tarball = builder.ensure(dist::Std {
            compiler: self.compiler,
            target: self.target
        }).expect("missing std");
        install_sh(builder, "std", self.compiler.stage, Some(self.target), &tarball);
    };
    Cargo, alias = "cargo", Self::should_build(_config), only_hosts: true, {
        let tarball = builder
            .ensure(dist::Cargo { compiler: self.compiler, target: self.target })
            .expect("missing cargo");
        install_sh(builder, "cargo", self.compiler.stage, Some(self.target), &tarball);
    };
    RustAnalyzer, alias = "rust-analyzer", Self::should_build(_config), only_hosts: true, {
        if let Some(tarball) =
            builder.ensure(dist::RustAnalyzer { compiler: self.compiler, target: self.target })
        {
            install_sh(builder, "rust-analyzer", self.compiler.stage, Some(self.target), &tarball);
        } else {
            builder.info(
                &format!("skipping Install rust-analyzer stage{} ({})", self.compiler.stage, self.target),
            );
        }
    };
    Clippy, alias = "clippy", Self::should_build(_config), only_hosts: true, {
        let tarball = builder
            .ensure(dist::Clippy { compiler: self.compiler, target: self.target })
            .expect("missing clippy");
        install_sh(builder, "clippy", self.compiler.stage, Some(self.target), &tarball);
    };
    Miri, alias = "miri", Self::should_build(_config), only_hosts: true, {
        if let Some(tarball) = builder.ensure(dist::Miri { compiler: self.compiler, target: self.target }) {
            install_sh(builder, "miri", self.compiler.stage, Some(self.target), &tarball);
        } else {
            // Miri is only available on nightly
            builder.info(
                &format!("skipping Install miri stage{} ({})", self.compiler.stage, self.target),
            );
        }
    };
    LlvmTools, alias = "llvm-tools", _config.llvm_tools_enabled && _config.llvm_enabled(_config.build), only_hosts: true, {
        if let Some(tarball) = builder.ensure(dist::LlvmTools { target: self.target }) {
            install_sh(builder, "llvm-tools", self.compiler.stage, Some(self.target), &tarball);
        } else {
            builder.info(
                &format!("skipping llvm-tools stage{} ({}): external LLVM", self.compiler.stage, self.target),
            );
        }
    };
    Rustfmt, alias = "rustfmt", Self::should_build(_config), only_hosts: true, {
        if let Some(tarball) = builder.ensure(dist::Rustfmt {
            compiler: self.compiler,
            target: self.target
        }) {
            install_sh(builder, "rustfmt", self.compiler.stage, Some(self.target), &tarball);
        } else {
            builder.info(
                &format!("skipping Install Rustfmt stage{} ({})", self.compiler.stage, self.target),
            );
        }
    };
    Rustc, path = "compiler/rustc", true, only_hosts: true, {
        let tarball = builder.ensure(dist::Rustc {
            compiler: builder.compiler(builder.top_stage, self.target),
        });
        install_sh(builder, "rustc", self.compiler.stage, Some(self.target), &tarball);
    };
    RustcCodegenCranelift, alias = "rustc-codegen-cranelift", Self::should_build(_config), only_hosts: true, {
        if let Some(tarball) = builder.ensure(dist::CodegenBackend {
            compiler: self.compiler,
            backend: "cranelift".to_string(),
        }) {
            install_sh(builder, "rustc-codegen-cranelift", self.compiler.stage, Some(self.target), &tarball);
        } else {
            builder.info(
                &format!("skipping Install CodegenBackend(\"cranelift\") stage{} ({})",
                         self.compiler.stage, self.target),
            );
        }
    };
    LlvmBitcodeLinker, alias = "llvm-bitcode-linker", Self::should_build(_config), only_hosts: true, {
        if let Some(tarball) = builder.ensure(dist::LlvmBitcodeLinker { compiler: self.compiler, target: self.target }) {
            install_sh(builder, "llvm-bitcode-linker", self.compiler.stage, Some(self.target), &tarball);
        } else {
            builder.info(
                &format!("skipping llvm-bitcode-linker stage{} ({})", self.compiler.stage, self.target),
            );
        }
    };
);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Src {
    pub stage: u32,
}

impl Step for Src {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let config = &run.builder.config;
        let cond = config.extended && config.tools.as_ref().is_none_or(|t| t.contains("src"));
        run.path("src").default_condition(cond)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Src { stage: run.builder.top_stage });
    }

    fn run(self, builder: &Builder<'_>) {
        let tarball = builder.ensure(dist::Src);
        install_sh(builder, "src", self.stage, None, &tarball);
    }
}
