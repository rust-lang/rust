use crate::utils::{get_gcc_path, get_os_name, rustc_version_info, split_args};
use std::collections::HashMap;
use std::env as std_env;

#[derive(Default)]
pub struct ConfigInfo {
    pub target: String,
    pub target_triple: String,
    pub host_triple: String,
    pub rustc_command: Vec<String>,
    pub run_in_vm: bool,
    pub cargo_target_dir: String,
    pub dylib_ext: String,
    pub sysroot_release_channel: bool,
    pub sysroot_panic_abort: bool,
}

impl ConfigInfo {
    /// Returns `true` if the argument was taken into account.
    pub fn parse_argument(
        &mut self,
        arg: &str,
        args: &mut impl Iterator<Item = String>,
    ) -> Result<bool, String> {
        match arg {
            "--target-triple" => match args.next() {
                Some(arg) if !arg.is_empty() => self.target_triple = arg.to_string(),
                _ => {
                    return Err(
                        "Expected a value after `--target-triple`, found nothing".to_string()
                    )
                }
            },
            "--target" => match args.next() {
                Some(arg) if !arg.is_empty() => self.target = arg.to_string(),
                _ => return Err("Expected a value after `--target`, found nothing".to_string()),
            },
            "--out-dir" => match args.next() {
                Some(arg) if !arg.is_empty() => {
                    // env.insert("CARGO_TARGET_DIR".to_string(), arg.to_string());
                    self.cargo_target_dir = arg.to_string();
                }
                _ => return Err("Expected a value after `--out-dir`, found nothing".to_string()),
            },
            "--release-sysroot" => self.sysroot_release_channel = true,
            "--sysroot-panic-abort" => self.sysroot_panic_abort = true,
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub fn setup(
        &mut self,
        env: &mut HashMap<String, String>,
        test_flags: &[String],
        gcc_path: Option<&str>,
    ) -> Result<(), String> {
        env.insert("CARGO_INCREMENTAL".to_string(), "0".to_string());

        let gcc_path = match gcc_path {
            Some(path) => path.to_string(),
            None => get_gcc_path()?,
        };
        env.insert("GCC_PATH".to_string(), gcc_path.clone());

        if self.cargo_target_dir.is_empty() {
            match env.get("CARGO_TARGET_DIR").filter(|dir| !dir.is_empty()) {
                Some(cargo_target_dir) => self.cargo_target_dir = cargo_target_dir.clone(),
                None => self.cargo_target_dir = "target/out".to_string(),
            }
        }

        let os_name = get_os_name()?;
        self.dylib_ext = match os_name.as_str() {
            "Linux" => "so",
            "Darwin" => "dylib",
            os => return Err(format!("unsupported OS `{}`", os)),
        }
        .to_string();
        let rustc = match env.get("RUSTC") {
            Some(r) if !r.is_empty() => r.to_string(),
            _ => "rustc".to_string(),
        };
        self.host_triple = rustc_version_info(Some(&rustc))?.host.unwrap_or_default();

        if !self.target_triple.is_empty() && self.target.is_empty() {
            self.target = self.target_triple.clone();
        }
        if self.target.is_empty() {
            self.target = self.host_triple.clone();
        }
        if self.target_triple.is_empty() {
            self.target_triple = self.host_triple.clone();
        }

        let mut linker = None;

        if self.host_triple != self.target_triple {
            if self.target_triple == "m68k-unknown-linux-gnu" {
                linker = Some("-Clinker=m68k-unknown-linux-gnu-gcc".to_string());
            } else if self.target_triple == "aarch64-unknown-linux-gnu" {
                // We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
                linker = Some("-Clinker=aarch64-linux-gnu-gcc".to_string());
            } else {
                return Err("Unknown non-native platform".to_string());
            }

            self.run_in_vm = true;
        }

        let current_dir =
            std_env::current_dir().map_err(|error| format!("`current_dir` failed: {:?}", error))?;
        let channel = if self.sysroot_release_channel {
            "release"
        } else if let Some(channel) = env.get("CHANNEL") {
            channel.as_str()
        } else {
            "debug"
        };

        let has_builtin_backend = env
            .get("BUILTIN_BACKEND")
            .map(|backend| !backend.is_empty())
            .unwrap_or(false);
        let cg_backend_path;

        let mut rustflags = Vec::new();
        if has_builtin_backend {
            // It means we're building inside the rustc testsuite, so some options need to be handled
            // a bit differently.
            cg_backend_path = "gcc".to_string();

            match env.get("RUSTC_SYSROOT") {
                Some(rustc_sysroot) if !rustc_sysroot.is_empty() => {
                    rustflags.extend_from_slice(&["--sysroot".to_string(), rustc_sysroot.clone()]);
                }
                _ => {}
            }
            rustflags.push("-Cpanic=abort".to_string());
        } else {
            cg_backend_path = current_dir
                .join("target")
                .join(channel)
                .join(&format!("librustc_codegen_gcc.{}", self.dylib_ext))
                .display()
                .to_string();
            let sysroot_path = current_dir.join("build_sysroot/sysroot");
            rustflags
                .extend_from_slice(&["--sysroot".to_string(), sysroot_path.display().to_string()]);
        };

        if let Some(cg_rustflags) = env.get("CG_RUSTFLAGS") {
            rustflags.extend_from_slice(&split_args(&cg_rustflags));
        }
        if let Some(linker) = linker {
            rustflags.push(linker.to_string());
        }
        rustflags.extend_from_slice(&[
            "-Csymbol-mangling-version=v0".to_string(),
            "-Cdebuginfo=2".to_string(),
            format!("-Zcodegen-backend={}", cg_backend_path),
        ]);

        // Since we don't support ThinLTO, disable LTO completely when not trying to do LTO.
        // TODO(antoyo): remove when we can handle ThinLTO.
        if !env.contains_key(&"FAT_LTO".to_string()) {
            rustflags.push("-Clto=off".to_string());
        }
        rustflags.extend_from_slice(test_flags);
        // FIXME(antoyo): remove once the atomic shim is gone
        if os_name == "Darwin" {
            rustflags.extend_from_slice(&[
                "-Clink-arg=-undefined".to_string(),
                "-Clink-arg=dynamic_lookup".to_string(),
            ]);
        }
        env.insert("RUSTFLAGS".to_string(), rustflags.join(" "));
        // display metadata load errors
        env.insert("RUSTC_LOG".to_string(), "warn".to_string());

        let sysroot = current_dir.join(&format!(
            "build_sysroot/sysroot/lib/rustlib/{}/lib",
            self.target_triple,
        ));
        let ld_library_path = format!(
            "{target}:{sysroot}:{gcc_path}",
            target = current_dir.join("target/out").display(),
            sysroot = sysroot.display(),
        );
        env.insert("LD_LIBRARY_PATH".to_string(), ld_library_path.clone());
        env.insert("DYLD_LIBRARY_PATH".to_string(), ld_library_path);

        // NOTE: To avoid the -fno-inline errors, use /opt/gcc/bin/gcc instead of cc.
        // To do so, add a symlink for cc to /opt/gcc/bin/gcc in our PATH.
        // Another option would be to add the following Rust flag: -Clinker=/opt/gcc/bin/gcc
        let path = std::env::var("PATH").unwrap_or_default();
        env.insert(
            "PATH".to_string(),
            format!(
                "/opt/gcc/bin:/opt/m68k-unknown-linux-gnu/bin{}{}",
                if path.is_empty() { "" } else { ":" },
                path
            ),
        );

        self.rustc_command = vec![rustc];
        self.rustc_command.extend_from_slice(&rustflags);
        self.rustc_command.extend_from_slice(&[
            "-L".to_string(),
            "crate=target/out".to_string(),
            "--out-dir".to_string(),
            self.cargo_target_dir.clone(),
        ]);

        if !env.contains_key("RUSTC_LOG") {
            env.insert("RUSTC_LOG".to_string(), "warn".to_string());
        }
        Ok(())
    }

    pub fn show_usage() {
        println!(
            "\
    --target [arg]         : Set the target to [arg]
    --target-triple [arg]  : Set the target triple to [arg]
    --out-dir              : Location where the files will be generated
    --release-sysroot      : Build sysroot in release mode
    --sysroot-panic-abort  : Build the sysroot without unwinding support."
        );
    }
}
