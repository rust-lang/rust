// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the install aspects of the compiler.
//!
//! This module is responsible for installing the standard library,
//! compiler, and documentation.

use std::env;
use std::fs;
use std::path::{Path, PathBuf, Component};
use std::process::Command;

use dist::{self, pkgname, sanitize_sh, tmpdir};

use builder::{Builder, ShouldRun, Step};
use cache::Interned;

pub fn install_docs(builder: &Builder, stage: u32, host: Interned<String>) {
    install_sh(builder, "docs", "rust-docs", stage, Some(host));
}

pub fn install_std(builder: &Builder, stage: u32) {
    for target in builder.build.config.target.iter() {
        install_sh(builder, "std", "rust-std", stage, Some(*target));
    }
}

pub fn install_cargo(builder: &Builder, stage: u32, host: Interned<String>) {
    install_sh(builder, "cargo", "cargo", stage, Some(host));
}

pub fn install_rls(builder: &Builder, stage: u32, host: Interned<String>) {
    install_sh(builder, "rls", "rls", stage, Some(host));
}

pub fn install_analysis(builder: &Builder, stage: u32, host: Interned<String>) {
    install_sh(builder, "analysis", "rust-analysis", stage, Some(host));
}

pub fn install_src(builder: &Builder, stage: u32) {
    install_sh(builder, "src", "rust-src", stage, None);
}
pub fn install_rustc(builder: &Builder, stage: u32, host: Interned<String>) {
    install_sh(builder, "rustc", "rustc", stage, Some(host));
}

fn install_sh(
    builder: &Builder,
    package: &str,
    name: &str,
    stage: u32,
    host: Option<Interned<String>>
) {
    let build = builder.build;
    println!("Install {} stage{} ({:?})", package, stage, host);

    let prefix_default = PathBuf::from("/usr/local");
    let sysconfdir_default = PathBuf::from("/etc");
    let docdir_default = PathBuf::from("share/doc/rust");
    let bindir_default = PathBuf::from("bin");
    let libdir_default = PathBuf::from("lib");
    let mandir_default = PathBuf::from("share/man");
    let prefix = build.config.prefix.as_ref().unwrap_or(&prefix_default);
    let sysconfdir = build.config.sysconfdir.as_ref().unwrap_or(&sysconfdir_default);
    let docdir = build.config.docdir.as_ref().unwrap_or(&docdir_default);
    let bindir = build.config.bindir.as_ref().unwrap_or(&bindir_default);
    let libdir = build.config.libdir.as_ref().unwrap_or(&libdir_default);
    let mandir = build.config.mandir.as_ref().unwrap_or(&mandir_default);

    let sysconfdir = prefix.join(sysconfdir);
    let docdir = prefix.join(docdir);
    let bindir = prefix.join(bindir);
    let libdir = prefix.join(libdir);
    let mandir = prefix.join(mandir);

    let destdir = env::var_os("DESTDIR").map(PathBuf::from);

    let prefix = add_destdir(&prefix, &destdir);
    let sysconfdir = add_destdir(&sysconfdir, &destdir);
    let docdir = add_destdir(&docdir, &destdir);
    let bindir = add_destdir(&bindir, &destdir);
    let libdir = add_destdir(&libdir, &destdir);
    let mandir = add_destdir(&mandir, &destdir);

    let empty_dir = build.out.join("tmp/empty_dir");

    t!(fs::create_dir_all(&empty_dir));
    let package_name = if let Some(host) = host {
        format!("{}-{}", pkgname(build, name), host)
    } else {
        pkgname(build, name)
    };

    let mut cmd = Command::new("sh");
    cmd.current_dir(&empty_dir)
        .arg(sanitize_sh(&tmpdir(build).join(&package_name).join("install.sh")))
        .arg(format!("--prefix={}", sanitize_sh(&prefix)))
        .arg(format!("--sysconfdir={}", sanitize_sh(&sysconfdir)))
        .arg(format!("--docdir={}", sanitize_sh(&docdir)))
        .arg(format!("--bindir={}", sanitize_sh(&bindir)))
        .arg(format!("--libdir={}", sanitize_sh(&libdir)))
        .arg(format!("--mandir={}", sanitize_sh(&mandir)))
        .arg("--disable-ldconfig");
    build.run(&mut cmd);
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
    (($sel:ident, $builder:ident),
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

        impl Step for $name {
            type Output = ();
            const DEFAULT: bool = true;
            const ONLY_BUILD_TARGETS: bool = true;
            const ONLY_HOSTS: bool = $only_hosts;
            $(const $c: bool = true;)*

            fn should_run(run: ShouldRun) -> ShouldRun {
                run.path($path)
            }

            fn make_run(
                $builder: &Builder,
                path: Option<&Path>,
                host: Interned<String>,
                target: Interned<String>,
            ) {
                if path.is_none() && !($default_cond) {
                    return;
                }
                $builder.ensure($name {
                    stage: $builder.top_stage,
                    target,
                    host,
                });
            }

            fn run($sel, $builder: &Builder) {
                $run_item
            }
        })+
    }
}

install!((self, builder),
    // rules.install("install-docs", "src/doc")
    //      .default(build.config.docs)
    //      .only_host_build(true)
    //      .dep(|s| s.name("dist-docs"))
    //      .run(move |s| install::Installer::new(build).install_docs(s.stage, s.target));
    Docs, "src/doc", builder.build.config.docs, only_hosts: false, {
        builder.ensure(dist::Docs { stage: self.stage, target: self.target });
        install_docs(builder, self.stage, self.target);
    };
    // rules.install("install-std", "src/libstd")
    //      .default(true)
    //      .only_host_build(true)
    //      .dep(|s| s.name("dist-std"))
    //      .run(move |s| install::Installer::new(build).install_std(s.stage));
    Std, "src/libstd", true, only_hosts: true, {
        builder.ensure(dist::Std {
            compiler: builder.compiler(self.stage, self.host),
            target: self.target
        });
        install_std(builder, self.stage);
    };
    // rules.install("install-cargo", "cargo")
    //      .default(build.config.extended)
    //      .host(true)
    //      .only_host_build(true)
    //      .dep(|s| s.name("dist-cargo"))
    //      .run(move |s| install::Installer::new(build).install_cargo(s.stage, s.target));
    Cargo, "cargo", builder.build.config.extended, only_hosts: true, {
        builder.ensure(dist::Cargo { stage: self.stage, target: self.target });
        install_cargo(builder, self.stage, self.target);
    };
    // rules.install("install-rls", "rls")
    //      .default(build.config.extended)
    //      .host(true)
    //      .only_host_build(true)
    //      .dep(|s| s.name("dist-rls"))
    //      .run(move |s| install::Installer::new(build).install_rls(s.stage, s.target));
    Rls, "rls", builder.build.config.extended, only_hosts: true, {
        builder.ensure(dist::Rls { stage: self.stage, target: self.target });
        install_rls(builder, self.stage, self.target);
    };
    // rules.install("install-analysis", "analysis")
    //      .default(build.config.extended)
    //      .only_host_build(true)
    //      .dep(|s| s.name("dist-analysis"))
    //      .run(move |s| install::Installer::new(build).install_analysis(s.stage, s.target));
    Analysis, "analysis", builder.build.config.extended, only_hosts: false, {
        builder.ensure(dist::Analysis {
            compiler: builder.compiler(self.stage, self.host),
            target: self.target
        });
        install_analysis(builder, self.stage, self.target);
    };
    // rules.install("install-src", "src")
    //      .default(build.config.extended)
    //      .host(true)
    //      .only_build(true)
    //      .only_host_build(true)
    //      .dep(|s| s.name("dist-src"))
    //      .run(move |s| install::Installer::new(build).install_src(s.stage));
    Src, "src", builder.build.config.extended, only_hosts: true, {
        builder.ensure(dist::Src);
        install_src(builder, self.stage);
    }, ONLY_BUILD;
    // rules.install("install-rustc", "src/librustc")
    //      .default(true)
    //      .host(true)
    //      .only_host_build(true)
    //      .dep(|s| s.name("dist-rustc"))
    //      .run(move |s| install::Installer::new(build).install_rustc(s.stage, s.target));
    Rustc, "src/librustc", builder.build.config.extended, only_hosts: true, {
        builder.ensure(dist::Rustc { stage: self.stage, target: self.target });
        install_rustc(builder, self.stage, self.target);
    };
);
