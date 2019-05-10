//! Implementation of the install aspects of the compiler.
//!
//! This module is responsible for installing the standard library,
//! compiler, and documentation.

use std::env;
use std::fs;
use std::path::{Path, PathBuf, Component};
use std::process::Command;

use build_helper::t;

use crate::dist::{self, pkgname, sanitize_sh, tmpdir};

use crate::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::cache::Interned;
use crate::config::Config;

pub fn install_docs(builder: &Builder<'_>, stage: u32, host: Interned<String>) {
    install_sh(builder, "docs", "rust-docs", stage, Some(host));
}

pub fn install_std(builder: &Builder<'_>, stage: u32, target: Interned<String>) {
    install_sh(builder, "std", "rust-std", stage, Some(target));
}

pub fn install_cargo(builder: &Builder<'_>, stage: u32, host: Interned<String>) {
    install_sh(builder, "cargo", "cargo", stage, Some(host));
}

pub fn install_rls(builder: &Builder<'_>, stage: u32, host: Interned<String>) {
    install_sh(builder, "rls", "rls", stage, Some(host));
}
pub fn install_clippy(builder: &Builder<'_>, stage: u32, host: Interned<String>) {
    install_sh(builder, "clippy", "clippy", stage, Some(host));
}
pub fn install_miri(builder: &Builder<'_>, stage: u32, host: Interned<String>) {
    install_sh(builder, "miri", "miri", stage, Some(host));
}

pub fn install_rustfmt(builder: &Builder<'_>, stage: u32, host: Interned<String>) {
    install_sh(builder, "rustfmt", "rustfmt", stage, Some(host));
}

pub fn install_analysis(builder: &Builder<'_>, stage: u32, host: Interned<String>) {
    install_sh(builder, "analysis", "rust-analysis", stage, Some(host));
}

pub fn install_src(builder: &Builder<'_>, stage: u32) {
    install_sh(builder, "src", "rust-src", stage, None);
}
pub fn install_rustc(builder: &Builder<'_>, stage: u32, host: Interned<String>) {
    install_sh(builder, "rustc", "rustc", stage, Some(host));
}

fn install_sh(
    builder: &Builder<'_>,
    package: &str,
    name: &str,
    stage: u32,
    host: Option<Interned<String>>
) {
    builder.info(&format!("Install {} stage{} ({:?})", package, stage, host));

    let prefix_default = PathBuf::from("/usr/local");
    let sysconfdir_default = PathBuf::from("/etc");
    let datadir_default = PathBuf::from("share");
    let docdir_default = datadir_default.join("doc/rust");
    let bindir_default = PathBuf::from("bin");
    let libdir_default = PathBuf::from("lib");
    let mandir_default = datadir_default.join("man");
    let prefix = builder.config.prefix.as_ref().map_or(prefix_default, |p| {
        fs::canonicalize(p).unwrap_or_else(|_| panic!("could not canonicalize {}", p.display()))
    });
    let sysconfdir = builder.config.sysconfdir.as_ref().unwrap_or(&sysconfdir_default);
    let datadir = builder.config.datadir.as_ref().unwrap_or(&datadir_default);
    let docdir = builder.config.docdir.as_ref().unwrap_or(&docdir_default);
    let bindir = builder.config.bindir.as_ref().unwrap_or(&bindir_default);
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

    let empty_dir = builder.out.join("tmp/empty_dir");

    t!(fs::create_dir_all(&empty_dir));
    let package_name = if let Some(host) = host {
        format!("{}-{}", pkgname(builder, name), host)
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
        match part {
            Component::Normal(s) => ret.push(s),
            _ => {}
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
            pub stage: u32,
            pub target: Interned<String>,
            pub host: Interned<String>,
        }

        impl $name {
            #[allow(dead_code)]
            fn should_build(config: &Config) -> bool {
                config.extended && config.tools.as_ref()
                    .map_or(true, |t| t.contains($path))
            }

            #[allow(dead_code)]
            fn should_install(builder: &Builder<'_>) -> bool {
                builder.config.tools.as_ref().map_or(false, |t| t.contains($path))
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
                    stage: run.builder.top_stage,
                    target: run.target,
                    host: run.builder.config.build,
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
        builder.ensure(dist::Docs { stage: self.stage, host: self.target });
        install_docs(builder, self.stage, self.target);
    };
    Std, "src/libstd", true, only_hosts: true, {
        for target in &builder.targets {
            builder.ensure(dist::Std {
                compiler: builder.compiler(self.stage, self.host),
                target: *target
            });
            install_std(builder, self.stage, *target);
        }
    };
    Cargo, "cargo", Self::should_build(_config), only_hosts: true, {
        builder.ensure(dist::Cargo { stage: self.stage, target: self.target });
        install_cargo(builder, self.stage, self.target);
    };
    Rls, "rls", Self::should_build(_config), only_hosts: true, {
        if builder.ensure(dist::Rls { stage: self.stage, target: self.target }).is_some() ||
            Self::should_install(builder) {
            install_rls(builder, self.stage, self.target);
        } else {
            builder.info(&format!("skipping Install RLS stage{} ({})", self.stage, self.target));
        }
    };
    Clippy, "clippy", Self::should_build(_config), only_hosts: true, {
        if builder.ensure(dist::Clippy { stage: self.stage, target: self.target }).is_some() ||
            Self::should_install(builder) {
            install_clippy(builder, self.stage, self.target);
        } else {
            builder.info(&format!("skipping Install clippy stage{} ({})", self.stage, self.target));
        }
    };
    Miri, "miri", Self::should_build(_config), only_hosts: true, {
        if builder.ensure(dist::Miri { stage: self.stage, target: self.target }).is_some() ||
            Self::should_install(builder) {
            install_miri(builder, self.stage, self.target);
        } else {
            builder.info(&format!("skipping Install miri stage{} ({})", self.stage, self.target));
        }
    };
    Rustfmt, "rustfmt", Self::should_build(_config), only_hosts: true, {
        if builder.ensure(dist::Rustfmt { stage: self.stage, target: self.target }).is_some() ||
            Self::should_install(builder) {
            install_rustfmt(builder, self.stage, self.target);
        } else {
            builder.info(
                &format!("skipping Install Rustfmt stage{} ({})", self.stage, self.target));
        }
    };
    Analysis, "analysis", Self::should_build(_config), only_hosts: false, {
        builder.ensure(dist::Analysis {
            compiler: builder.compiler(self.stage, self.host),
            target: self.target
        });
        install_analysis(builder, self.stage, self.target);
    };
    Rustc, "src/librustc", true, only_hosts: true, {
        builder.ensure(dist::Rustc {
            compiler: builder.compiler(self.stage, self.target),
        });
        install_rustc(builder, self.stage, self.target);
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
        let cond = config.extended &&
            config.tools.as_ref().map_or(true, |t| t.contains("src"));
        run.path("src").default_condition(cond)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Src {
            stage: run.builder.top_stage,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(dist::Src);
        install_src(builder, self.stage);
    }
}
