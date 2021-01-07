//! Implementation of the install aspects of the compiler.
//!
//! This module is responsible for installing the standard library,
//! compiler, and documentation.

use std::env;
use std::fs;
use std::path::{Component, PathBuf};
use std::process::Command;

use build_helper::t;

use crate::dist::{self, sanitize_sh};
use crate::tarball::GeneratedTarball;
use crate::Compiler;

use crate::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::config::{Config, TargetSelection};

fn install_sh(
    builder: &Builder<'_>,
    package: &str,
    stage: u32,
    host: Option<TargetSelection>,
    tarball: &GeneratedTarball,
) {
    builder.info(&format!("Install {} stage{} ({:?})", package, stage, host));

    let prefix = default_path(&builder.config.prefix, "/usr/local");
    let sysconfdir = prefix.join(default_path(&builder.config.sysconfdir, "/etc"));
    let datadir = prefix.join(default_path(&builder.config.datadir, "share"));
    let docdir = prefix.join(default_path(&builder.config.docdir, "share/doc"));
    let mandir = prefix.join(default_path(&builder.config.mandir, "share/man"));
    let libdir = prefix.join(default_path(&builder.config.libdir, "lib"));
    let bindir = prefix.join(&builder.config.bindir); // Default in config.rs

    let empty_dir = builder.out.join("tmp/empty_dir");
    t!(fs::create_dir_all(&empty_dir));

    let mut cmd = Command::new("sh");
    cmd.current_dir(&empty_dir)
        .arg(sanitize_sh(&tarball.decompressed_output().join("install.sh")))
        .arg(format!("--prefix={}", prepare_dir(prefix)))
        .arg(format!("--sysconfdir={}", prepare_dir(sysconfdir)))
        .arg(format!("--datadir={}", prepare_dir(datadir)))
        .arg(format!("--docdir={}", prepare_dir(docdir)))
        .arg(format!("--bindir={}", prepare_dir(bindir)))
        .arg(format!("--libdir={}", prepare_dir(libdir)))
        .arg(format!("--mandir={}", prepare_dir(mandir)))
        .arg("--disable-ldconfig");
    builder.run(&mut cmd);
    t!(fs::remove_dir_all(&empty_dir));
}

fn default_path(config: &Option<PathBuf>, default: &str) -> PathBuf {
    PathBuf::from(config.as_ref().cloned().unwrap_or_else(|| PathBuf::from(default)))
}

fn prepare_dir(mut path: PathBuf) -> String {
    // The DESTDIR environment variable is a standard way to install software in a subdirectory
    // while keeping the original directory structure, even if the prefix or other directories
    // contain absolute paths.
    //
    // More information on the environment variable is available here:
    // https://www.gnu.org/prep/standards/html_node/DESTDIR.html
    if let Some(destdir) = env::var_os("DESTDIR").map(PathBuf::from) {
        let without_destdir = path.clone();
        path = destdir;
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

    sanitize_sh(&path)
}

macro_rules! install {
    (($sel:ident, $builder:ident, $_config:ident),
       $($name:ident,
       $path:expr,
       $default_cond:expr,
       only_hosts: $only_hosts:expr,
       $run_item:block $(, $c:ident)*;)+) => {
        $(
            #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            pub compiler: Compiler,
            pub target: TargetSelection,
        }

        impl $name {
            #[allow(dead_code)]
            fn should_build(config: &Config) -> bool {
                config.extended && config.tools.as_ref()
                    .map_or(true, |t| t.contains($path))
            }
        }

        impl Step for $name {
            type Output = ();
            const DEFAULT: bool = true;
            const ONLY_HOSTS: bool = $only_hosts;
            $(const $c: bool = true;)*

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                let $_config = &run.builder.config;
                run.path($path).default_condition($default_cond)
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
    Docs, "src/doc", _config.docs, only_hosts: false, {
        let tarball = builder.ensure(dist::Docs { host: self.target }).expect("missing docs");
        install_sh(builder, "docs", self.compiler.stage, Some(self.target), &tarball);
    };
    Std, "library/std", true, only_hosts: false, {
        for target in &builder.targets {
            let tarball = builder.ensure(dist::Std {
                compiler: self.compiler,
                target: *target
            }).expect("missing std");
            install_sh(builder, "std", self.compiler.stage, Some(*target), &tarball);
        }
    };
    Cargo, "cargo", Self::should_build(_config), only_hosts: true, {
        let tarball = builder.ensure(dist::Cargo { compiler: self.compiler, target: self.target });
        install_sh(builder, "cargo", self.compiler.stage, Some(self.target), &tarball);
    };
    Rls, "rls", Self::should_build(_config), only_hosts: true, {
        if let Some(tarball) = builder.ensure(dist::Rls { compiler: self.compiler, target: self.target }) {
            install_sh(builder, "rls", self.compiler.stage, Some(self.target), &tarball);
        } else {
            builder.info(
                &format!("skipping Install RLS stage{} ({})", self.compiler.stage, self.target),
            );
        }
    };
    RustAnalyzer, "rust-analyzer", Self::should_build(_config), only_hosts: true, {
        let tarball = builder
            .ensure(dist::RustAnalyzer { compiler: self.compiler, target: self.target })
            .expect("missing rust-analyzer");
        install_sh(builder, "rust-analyzer", self.compiler.stage, Some(self.target), &tarball);
    };
    Clippy, "clippy", Self::should_build(_config), only_hosts: true, {
        let tarball = builder.ensure(dist::Clippy { compiler: self.compiler, target: self.target });
        install_sh(builder, "clippy", self.compiler.stage, Some(self.target), &tarball);
    };
    Miri, "miri", Self::should_build(_config), only_hosts: true, {
        if let Some(tarball) = builder.ensure(dist::Miri { compiler: self.compiler, target: self.target }) {
            install_sh(builder, "miri", self.compiler.stage, Some(self.target), &tarball);
        } else {
            builder.info(
                &format!("skipping Install miri stage{} ({})", self.compiler.stage, self.target),
            );
        }
    };
    Rustfmt, "rustfmt", Self::should_build(_config), only_hosts: true, {
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
    Analysis, "analysis", Self::should_build(_config), only_hosts: false, {
        let tarball = builder.ensure(dist::Analysis {
            // Find the actual compiler (handling the full bootstrap option) which
            // produced the save-analysis data because that data isn't copied
            // through the sysroot uplifting.
            compiler: builder.compiler_for(builder.top_stage, builder.config.build, self.target),
            target: self.target
        }).expect("missing analysis");
        install_sh(builder, "analysis", self.compiler.stage, Some(self.target), &tarball);
    };
    Rustc, "src/librustc", true, only_hosts: true, {
        let tarball = builder.ensure(dist::Rustc {
            compiler: builder.compiler(builder.top_stage, self.target),
        });
        install_sh(builder, "rustc", self.compiler.stage, Some(self.target), &tarball);
    };
);

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Src {
    pub stage: u32,
}

impl Step for Src {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let config = &run.builder.config;
        let cond = config.extended && config.tools.as_ref().map_or(true, |t| t.contains("src"));
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
