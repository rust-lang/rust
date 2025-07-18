use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::{env as std_env, fs};

use boml::Toml;
use boml::types::TomlValue;

use crate::utils::{
    create_dir, create_symlink, get_os_name, get_sysroot_dir, run_command_with_output,
    rustc_version_info, split_args,
};

#[derive(Default, PartialEq, Eq, Clone, Copy, Debug)]
pub enum Channel {
    #[default]
    Debug,
    Release,
}

impl Channel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

fn failed_config_parsing(config_file: &Path, err: &str) -> Result<ConfigFile, String> {
    Err(format!("Failed to parse `{}`: {}", config_file.display(), err))
}

#[derive(Default)]
pub struct ConfigFile {
    gcc_path: Option<String>,
    download_gccjit: Option<bool>,
}

impl ConfigFile {
    pub fn new(config_file: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(config_file).map_err(|_| {
            format!(
                "Failed to read `{}`. Take a look at `Readme.md` to see how to set up the project",
                config_file.display(),
            )
        })?;
        let toml = Toml::parse(&content).map_err(|err| {
            format!("Error occurred around `{}`: {:?}", &content[err.start..=err.end], err.kind)
        })?;
        let mut config = Self::default();
        for (key, value) in toml.iter() {
            match (key, value) {
                ("gcc-path", TomlValue::String(value)) => {
                    config.gcc_path = Some(value.as_str().to_string())
                }
                ("gcc-path", _) => {
                    return failed_config_parsing(config_file, "Expected a string for `gcc-path`");
                }
                ("download-gccjit", TomlValue::Boolean(value)) => {
                    config.download_gccjit = Some(*value)
                }
                ("download-gccjit", _) => {
                    return failed_config_parsing(
                        config_file,
                        "Expected a boolean for `download-gccjit`",
                    );
                }
                _ => return failed_config_parsing(config_file, &format!("Unknown key `{key}`")),
            }
        }
        match (config.gcc_path.as_mut(), config.download_gccjit) {
            (None, None | Some(false)) => {
                return failed_config_parsing(
                    config_file,
                    "At least one of `gcc-path` or `download-gccjit` value must be set",
                );
            }
            (Some(_), Some(true)) => {
                println!(
                    "WARNING: both `gcc-path` and `download-gccjit` arguments are used, \
                    ignoring `gcc-path`"
                );
            }
            (Some(gcc_path), _) => {
                let path = Path::new(gcc_path);
                *gcc_path = path
                    .canonicalize()
                    .map_err(|err| format!("Failed to get absolute path of `{gcc_path}`: {err:?}"))?
                    .display()
                    .to_string();
            }
            _ => {}
        }
        Ok(config)
    }
}

#[derive(Default, Debug, Clone)]
pub struct ConfigInfo {
    pub target: String,
    pub target_triple: String,
    pub host_triple: String,
    pub rustc_command: Vec<String>,
    pub run_in_vm: bool,
    pub cargo_target_dir: String,
    pub dylib_ext: String,
    pub sysroot_release_channel: bool,
    pub channel: Channel,
    pub sysroot_panic_abort: bool,
    pub cg_backend_path: String,
    pub sysroot_path: String,
    pub gcc_path: Option<String>,
    config_file: Option<String>,
    // This is used in particular in rust compiler bootstrap because it doesn't run at the root
    // of the `cg_gcc` folder, making it complicated for us to get access to local files we need
    // like `libgccjit.version` or `config.toml`.
    cg_gcc_path: Option<PathBuf>,
    // Needed for the `info` command which doesn't want to actually download the lib if needed,
    // just to set the `gcc_path` field to display it.
    pub no_download: bool,
    pub no_default_features: bool,
    pub backend: Option<String>,
    pub features: Vec<String>,
}

impl ConfigInfo {
    /// Returns `true` if the argument was taken into account.
    pub fn parse_argument(
        &mut self,
        arg: &str,
        args: &mut impl Iterator<Item = String>,
    ) -> Result<bool, String> {
        match arg {
            "--features" => {
                if let Some(arg) = args.next() {
                    self.features.push(arg);
                } else {
                    return Err("Expected a value after `--features`, found nothing".to_string());
                }
            }
            "--target" => {
                if let Some(arg) = args.next() {
                    self.target = arg;
                } else {
                    return Err("Expected a value after `--target`, found nothing".to_string());
                }
            }
            "--target-triple" => match args.next() {
                Some(arg) if !arg.is_empty() => self.target_triple = arg.to_string(),
                _ => {
                    return Err(
                        "Expected a value after `--target-triple`, found nothing".to_string()
                    );
                }
            },
            "--out-dir" => match args.next() {
                Some(arg) if !arg.is_empty() => {
                    self.cargo_target_dir = arg.to_string();
                }
                _ => return Err("Expected a value after `--out-dir`, found nothing".to_string()),
            },
            "--config-file" => match args.next() {
                Some(arg) if !arg.is_empty() => {
                    self.config_file = Some(arg.to_string());
                }
                _ => {
                    return Err("Expected a value after `--config-file`, found nothing".to_string());
                }
            },
            "--release-sysroot" => self.sysroot_release_channel = true,
            "--release" => self.channel = Channel::Release,
            "--sysroot-panic-abort" => self.sysroot_panic_abort = true,
            "--gcc-path" => match args.next() {
                Some(arg) if !arg.is_empty() => {
                    self.gcc_path = Some(arg);
                }
                _ => {
                    return Err("Expected a value after `--gcc-path`, found nothing".to_string());
                }
            },
            "--cg_gcc-path" => match args.next() {
                Some(arg) if !arg.is_empty() => {
                    self.cg_gcc_path = Some(arg.into());
                }
                _ => {
                    return Err("Expected a value after `--cg_gcc-path`, found nothing".to_string());
                }
            },
            "--use-backend" => match args.next() {
                Some(backend) if !backend.is_empty() => self.backend = Some(backend),
                _ => return Err("Expected an argument after `--use-backend`, found nothing".into()),
            },
            "--no-default-features" => self.no_default_features = true,
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub fn rustc_command_vec(&self) -> Vec<&dyn AsRef<OsStr>> {
        let mut command: Vec<&dyn AsRef<OsStr>> = Vec::with_capacity(self.rustc_command.len());
        for arg in self.rustc_command.iter() {
            command.push(arg);
        }
        command
    }

    pub fn get_gcc_commit(&self) -> Result<String, String> {
        let commit_hash_file = self.compute_path("libgccjit.version");
        let content = fs::read_to_string(&commit_hash_file).map_err(|_| {
            format!(
                "Failed to read `{}`. Take a look at `Readme.md` to see how to set up the project",
                commit_hash_file.display(),
            )
        })?;
        let commit = content.trim();
        // This is a very simple check to ensure this is not a path. For the rest, it'll just fail
        // when trying to download the file so we should be fine.
        if commit.contains('/') || commit.contains('\\') {
            return Err(format!(
                "{}: invalid commit hash `{}`",
                commit_hash_file.display(),
                commit,
            ));
        }
        Ok(commit.to_string())
    }

    fn download_gccjit_if_needed(&mut self) -> Result<(), String> {
        let output_dir = Path::new(crate::BUILD_DIR).join("libgccjit");
        let commit = self.get_gcc_commit()?;

        let output_dir = output_dir.join(&commit);
        if !output_dir.is_dir() {
            create_dir(&output_dir)?;
        }
        let output_dir = output_dir.canonicalize().map_err(|err| {
            format!("Failed to get absolute path of `{}`: {:?}", output_dir.display(), err)
        })?;

        let libgccjit_so_name = "libgccjit.so";
        let libgccjit_so = output_dir.join(libgccjit_so_name);
        if !libgccjit_so.is_file() && !self.no_download {
            // Download time!
            let tempfile_name = format!("{libgccjit_so_name}.download");
            let tempfile = output_dir.join(&tempfile_name);
            let is_in_ci = std::env::var("GITHUB_ACTIONS").is_ok();

            download_gccjit(&commit, &output_dir, tempfile_name, !is_in_ci)?;

            let libgccjit_so = output_dir.join(libgccjit_so_name);
            // If we reach this point, it means the file was correctly downloaded, so let's
            // rename it!
            std::fs::rename(&tempfile, &libgccjit_so).map_err(|err| {
                format!(
                    "Failed to rename `{}` into `{}`: {:?}",
                    tempfile.display(),
                    libgccjit_so.display(),
                    err,
                )
            })?;

            println!("Downloaded libgccjit.so version {commit} successfully!");
            // We need to create a link named `libgccjit.so.0` because that's what the linker is
            // looking for.
            create_symlink(&libgccjit_so, output_dir.join(format!("{libgccjit_so_name}.0")))?;
        }

        let gcc_path = output_dir.display().to_string();
        println!("Using `{gcc_path}` as path for libgccjit");
        self.gcc_path = Some(gcc_path);
        Ok(())
    }

    pub fn compute_path<P: AsRef<Path>>(&self, other: P) -> PathBuf {
        match self.cg_gcc_path {
            Some(ref path) => path.join(other),
            None => PathBuf::new().join(other),
        }
    }

    pub fn setup_gcc_path(&mut self) -> Result<(), String> {
        // If the user used the `--gcc-path` option, no need to look at `config.toml` content
        // since we already have everything we need.
        if let Some(gcc_path) = &self.gcc_path {
            println!(
                "`--gcc-path` was provided, ignoring config file. Using `{gcc_path}` as path for libgccjit"
            );
            return Ok(());
        }
        let config_file = match self.config_file.as_deref() {
            Some(config_file) => config_file.into(),
            None => self.compute_path("config.toml"),
        };
        let ConfigFile { gcc_path, download_gccjit } = ConfigFile::new(&config_file)?;

        if let Some(true) = download_gccjit {
            self.download_gccjit_if_needed()?;
            return Ok(());
        }
        let Some(gcc_path) = gcc_path else {
            return Err(format!("missing `gcc-path` value from `{}`", config_file.display()));
        };
        println!(
            "GCC path retrieved from `{}`. Using `{}` as path for libgccjit",
            config_file.display(),
            gcc_path
        );
        self.gcc_path = Some(gcc_path);
        Ok(())
    }

    pub fn setup(
        &mut self,
        env: &mut HashMap<String, String>,
        use_system_gcc: bool,
    ) -> Result<(), String> {
        env.insert("CARGO_INCREMENTAL".to_string(), "0".to_string());

        let gcc_path = if !use_system_gcc {
            if self.gcc_path.is_none() {
                self.setup_gcc_path()?;
            }
            self.gcc_path.clone().expect(
                "The config module should have emitted an error if the GCC path wasn't provided",
            )
        } else {
            String::new()
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
            os => return Err(format!("unsupported OS `{os}`")),
        }
        .to_string();
        let rustc = match env.get("RUSTC") {
            Some(r) if !r.is_empty() => r.to_string(),
            _ => "rustc".to_string(),
        };
        self.host_triple = match rustc_version_info(Some(&rustc))?.host {
            Some(host) => host,
            None => return Err("no host found".to_string()),
        };

        if self.target_triple.is_empty()
            && let Some(overwrite) = env.get("OVERWRITE_TARGET_TRIPLE")
        {
            self.target_triple = overwrite.clone();
        }
        if self.target_triple.is_empty() {
            self.target_triple = self.host_triple.clone();
        }
        if self.target.is_empty() && !self.target_triple.is_empty() {
            self.target = self.target_triple.clone();
        }

        let mut linker = None;

        if self.host_triple != self.target_triple {
            if self.target_triple.is_empty() {
                return Err("Unknown non-native platform".to_string());
            }
            linker = Some(format!("-Clinker={}-gcc", self.target_triple));
            self.run_in_vm = true;
        }

        let current_dir =
            std_env::current_dir().map_err(|error| format!("`current_dir` failed: {error:?}"))?;
        let channel = if self.channel == Channel::Release {
            "release"
        } else if let Some(channel) = env.get("CHANNEL") {
            channel.as_str()
        } else {
            "debug"
        };

        let mut rustflags = Vec::new();
        self.cg_backend_path = current_dir
            .join("target")
            .join(channel)
            .join(format!("librustc_codegen_gcc.{}", self.dylib_ext))
            .display()
            .to_string();
        self.sysroot_path =
            current_dir.join(get_sysroot_dir()).join("sysroot").display().to_string();
        if let Some(backend) = &self.backend {
            // This option is only used in the rust compiler testsuite. The sysroot is handled
            // by its build system directly so no need to set it ourselves.
            rustflags.push(format!("-Zcodegen-backend={backend}"));
        } else {
            rustflags.extend_from_slice(&[
                "--sysroot".to_string(),
                self.sysroot_path.clone(),
                format!("-Zcodegen-backend={}", self.cg_backend_path),
            ]);
        }

        // This environment variable is useful in case we want to change options of rustc commands.
        // We have a different environment variable than RUSTFLAGS to make sure those flags are
        // only sent to rustc_codegen_gcc and not the LLVM backend.
        if let Some(cg_rustflags) = env.get("CG_RUSTFLAGS") {
            rustflags.extend_from_slice(&split_args(cg_rustflags)?);
        }
        if let Some(test_flags) = env.get("TEST_FLAGS") {
            rustflags.extend_from_slice(&split_args(test_flags)?);
        }

        if let Some(linker) = linker {
            rustflags.push(linker.to_string());
        }

        if self.no_default_features {
            rustflags.push("-Csymbol-mangling-version=v0".to_string());
        }

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

        let sysroot = current_dir
            .join(get_sysroot_dir())
            .join(format!("sysroot/lib/rustlib/{}/lib", self.target_triple));
        let ld_library_path = format!(
            "{target}:{sysroot}:{gcc_path}",
            target = self.cargo_target_dir,
            sysroot = sysroot.display(),
            gcc_path = gcc_path,
        );
        env.insert("LIBRARY_PATH".to_string(), ld_library_path.clone());
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
            format!("crate={}", self.cargo_target_dir),
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
    --features [arg]       : Add a new feature [arg]
    --target-triple [arg]  : Set the target triple to [arg]
    --target [arg]         : Set the target to [arg]
    --out-dir              : Location where the files will be generated
    --release              : Build in release mode
    --release-sysroot      : Build sysroot in release mode
    --sysroot-panic-abort  : Build the sysroot without unwinding support
    --config-file          : Location of the config file to be used
    --gcc-path             : Location of the GCC root folder
    --cg_gcc-path          : Location of the rustc_codegen_gcc root folder (used
                             when ran from another directory)
    --no-default-features  : Add `--no-default-features` flag to cargo commands
    --use-backend          : Useful only for rustc testsuite"
        );
    }
}

fn download_gccjit(
    commit: &str,
    output_dir: &Path,
    tempfile_name: String,
    with_progress_bar: bool,
) -> Result<(), String> {
    let url = if std::env::consts::OS == "linux" && std::env::consts::ARCH == "x86_64" {
        format!("https://github.com/rust-lang/gcc/releases/download/master-{commit}/libgccjit.so")
    } else {
        eprintln!(
            "\
Pre-compiled libgccjit.so not available for this os or architecture.
Please compile it yourself and update the `config.toml` file
to `download-gccjit = false` and set `gcc-path` to the appropriate directory."
        );
        return Err(String::from(
            "no appropriate pre-compiled libgccjit.so available for download",
        ));
    };

    println!("Downloading `{url}`...");

    // Try curl. If that fails and we are on windows, fallback to PowerShell.
    let mut ret = run_command_with_output(
        &[
            &"curl",
            &"--speed-time",
            &"30",
            &"--speed-limit",
            &"10", // timeout if speed is < 10 bytes/sec for > 30 seconds
            &"--connect-timeout",
            &"30", // timeout if cannot connect within 30 seconds
            &"-o",
            &tempfile_name,
            &"--retry",
            &"3",
            &"-SRfL",
            if with_progress_bar { &"--progress-bar" } else { &"-s" },
            &url.as_str(),
        ],
        Some(output_dir),
    );
    if ret.is_err() && cfg!(windows) {
        eprintln!("Fallback to PowerShell");
        ret = run_command_with_output(
            &[
                &"PowerShell.exe",
                &"/nologo",
                &"-Command",
                &"[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;",
                &format!(
                    "(New-Object System.Net.WebClient).DownloadFile('{url}', '{tempfile_name}')",
                )
                .as_str(),
            ],
            Some(output_dir),
        );
    }
    ret
}
