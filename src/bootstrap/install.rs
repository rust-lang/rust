//! Implementation of the install aspects of the compiler.
//!
//! This module is responsible for installing the standard library,
//! compiler, and documentation.

use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use build_helper::t;

use crate::dist::{self, pkgname, sanitize_sh, tmpdir};
use crate::Compiler;

use crate::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::config::{Config, TargetSelection};

pub fn install_docs(builder: &Builder<'_>, stage: u32, host: TargetSelection) {
    install_sh(builder, "docs", "rust-docs", stage, Some(host));
}

pub fn install_std(builder: &Builder<'_>, stage: u32, target: TargetSelection) {
    install_sh(builder, "std", "rust-std", stage, Some(target));
}

pub fn install_cargo(builder: &Builder<'_>, stage: u32, host: TargetSelection) {
    install_sh(builder, "cargo", "cargo", stage, Some(host));
}

pub fn install_rls(builder: &Builder<'_>, stage: u32, host: TargetSelection) {
    install_sh(builder, "rls", "rls", stage, Some(host));
}

pub fn install_rust_analyzer(builder: &Builder<'_>, stage: u32, host: TargetSelection) {
    install_sh(builder, "rust-analyzer", "rust-analyzer", stage, Some(host));
}

pub fn install_clippy(builder: &Builder<'_>, stage: u32, host: TargetSelection) {
    install_sh(builder, "clippy", "clippy", stage, Some(host));
}
pub fn install_miri(builder: &Builder<'_>, stage: u32, host: TargetSelection) {
    install_sh(builder, "miri", "miri", stage, Some(host));
}

pub fn install_rustfmt(builder: &Builder<'_>, stage: u32, host: TargetSelection) {
    install_sh(builder, "rustfmt", "rustfmt", stage, Some(host));
}

pub fn install_analysis(builder: &Builder<'_>, stage: u32, host: TargetSelection) {
    install_sh(builder, "analysis", "rust-analysis", stage, Some(host));
}

pub fn install_src(builder: &Builder<'_>, stage: u32) {
    install_sh(builder, "src", "rust-src", stage, None);
}
pub fn install_rustc(builder: &Builder<'_>, stage: u32, host: TargetSelection) {
    install_sh(builder, "rustc", "rustc", stage, Some(host));
}

fn install_sh(
    builder: &Builder<'_>,
    package: &str,
    name: &str,
    stage: u32,
    host: Option<TargetSelection>,
) {
    builder.info(&format!("Install {} stage{} ({:?})", package, stage, host));

    let prefix_default = PathBuf::from("/usr/local");
    let sysconfdir_default = PathBuf::from("/etc");
    let datadir_default = PathBuf::from("share");
    let docdir_default = datadir_default.join("doc/rust");
    let libdir_default = PathBuf::from("lib");
    let mandir_default = datadir_default.join("man");
    let prefix = builder.config.prefix.as_ref().unwrap_or(&prefix_default);
    let sysconfdir = builder.config.sysconfdir.as_ref().unwrap_or(&sysconfdir_default);
    let datadir = builder.config.datadir.as_ref().unwrap_or(&datadir_default);
    let docdir = builder.config.docdir.as_ref().unwrap_or(&docdir_default);
    let bindir = &builder.config.bindir;
    let libdir = builder.config.libdir.as_ref().unwrap_or(&libdir_default);
    let mandir = builder.config.mandir.as_ref().unwrap_or(&mandir_default);

    let sysconfdir = prefix.join(sysconfdir);
    let datadir = prefix.join(datadir);
    let docdir = prefix.join(docdir);
    let bindir = prefix.join(bindir);
    let libdir = prefix.join(libdir);
    let mandir = prefix.join(mandir);

    let destdir = env::var_os("DESTDIR").map(PathBuf::from);

    let prefix = add_destdir(&prefix, &destdir);
    let sysconfdir = add_destdir(&sysconfdir, &destdir);
    let datadir = add_destdir(&datadir, &destdir);
    let docdir = add_destdir(&docdir, &destdir);
    let bindir = add_destdir(&bindir, &destdir);
    let libdir = add_destdir(&libdir, &destdir);
    let mandir = add_destdir(&mandir, &destdir);

    let prefix = {
        fs::create_dir_all(&prefix)
            .unwrap_or_else(|err| panic!("could not create {}: {}", prefix.display(), err));
        fs::canonicalize(&prefix)
            .unwrap_or_else(|err| panic!("could not canonicalize {}: {}", prefix.display(), err))
    };

    let empty_dir = builder.out.join("tmp/empty_dir");

    t!(fs::create_dir_all(&empty_dir));
    let package_name = if let Some(host) = host {
        format!("{}-{}", pkgname(builder, name), host.triple)
    } else {
        pkgname(builder, name)
    };

    let mut cmd = Command::new("sh");
    cmd.current_dir(&empty_dir)
        .arg(sanitize_sh(&tmpdir(builder).join(&package_name).join("install.sh")))
        .arg(format!("--prefix={}", sanitize_sh(&prefix)))
        .arg(format!("--sysconfdir={}", sanitize_sh(&sysconfdir)))
        .arg(format!("--datadir={}", sanitize_sh(&datadir)))
        .arg(format!("--docdir={}", sanitize_sh(&docdir)))
        .arg(format!("--bindir={}", sanitize_sh(&bindir)))
        .arg(format!("--libdir={}", sanitize_sh(&libdir)))
        .arg(format!("--mandir={}", sanitize_sh(&mandir)))
        .arg("--disable-ldconfig");
    builder.run(&mut cmd);
    t!(fs::remove_dir_all(&empty_dir));
}

fn add_destdir(path: &Path, destdir: &Option<PathBuf>) -> PathBuf {
    let mut ret = match *destdir {
        Some(ref dest) => dest.clone(),
        None => return path.to_path_buf(),
    };
    for part in path.components() {
        if let Component::Normal(s) = part {
            ret.push(s)
        }
    }
    ret
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
        builder.ensure(dist::Docs { host: self.target });
        install_docs(builder, self.compiler.stage, self.target);
    };
    Std, "library/std", true, only_hosts: false, {
        for target in &builder.targets {
            builder.ensure(dist::Std {
                compiler: self.compiler,
                target: *target
            });
            install_std(builder, self.compiler.stage, *target);
        }
    };
    Cargo, "cargo", Self::should_build(_config), only_hosts: true, {
        builder.ensure(dist::Cargo { compiler: self.compiler, target: self.target });
        install_cargo(builder, self.compiler.stage, self.target);
    };
    Rls, "rls", Self::should_build(_config), only_hosts: true, {
        if builder.ensure(dist::Rls { compiler: self.compiler, target: self.target }).is_some() {
            install_rls(builder, self.compiler.stage, self.target);
        } else {
            builder.info(
                &format!("skipping Install RLS stage{} ({})", self.compiler.stage, self.target),
            );
        }
    };
    RustAnalyzer, "rust-analyzer", Self::should_build(_config), only_hosts: true, {
        builder.ensure(dist::RustAnalyzer { compiler: self.compiler, target: self.target });
        install_rust_analyzer(builder, self.compiler.stage, self.target);
    };
    Clippy, "clippy", Self::should_build(_config), only_hosts: true, {
        builder.ensure(dist::Clippy { compiler: self.compiler, target: self.target });
        install_clippy(builder, self.compiler.stage, self.target);
    };
    Miri, "miri", Self::should_build(_config), only_hosts: true, {
        if builder.ensure(dist::Miri { compiler: self.compiler, target: self.target }).is_some() {
            install_miri(builder, self.compiler.stage, self.target);
        } else {
            builder.info(
                &format!("skipping Install miri stage{} ({})", self.compiler.stage, self.target),
            );
        }
    };
    Rustfmt, "rustfmt", Self::should_build(_config), only_hosts: true, {
        if builder.ensure(dist::Rustfmt {
            compiler: self.compiler,
            target: self.target
        }).is_some() {
            install_rustfmt(builder, self.compiler.stage, self.target);
        } else {
            builder.info(
                &format!("skipping Install Rustfmt stage{} ({})", self.compiler.stage, self.target),
            );
        }
    };
    Analysis, "analysis", Self::should_build(_config), only_hosts: false, {
        builder.ensure(dist::Analysis {
            // Find the actual compiler (handling the full bootstrap option) which
            // produced the save-analysis data because that data isn't copied
            // through the sysroot uplifting.
            compiler: builder.compiler_for(builder.top_stage, builder.config.build, self.target),
            target: self.target
        });
        install_analysis(builder, self.compiler.stage, self.target);
    };
    Rustc, "src/librustc", true, only_hosts: true, {
        builder.ensure(dist::Rustc {
            compiler: builder.compiler(builder.top_stage, self.target),
        });
        install_rustc(builder, self.compiler.stage, self.target);
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
        builder.ensure(dist::Src);
        install_src(builder, self.stage);
    }
}
