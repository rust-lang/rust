use std::path::PathBuf;

use anyhow::Context;

use crate::{Optimization, Target};

#[derive(Debug)]
pub struct Session {
    target: Target,
    cpu: Option<String>,
    feature: Option<String>,
    symbols: Vec<String>,

    /// A file that `llvm-link` supports, like a bitcode file or an archive.
    files: Vec<PathBuf>,

    // Output files
    link_path: PathBuf,
    opt_path: PathBuf,
    sym_path: PathBuf,
    out_path: PathBuf,
}

impl Session {
    pub fn new(
        target: crate::Target,
        cpu: Option<String>,
        feature: Option<String>,
        out_path: PathBuf,
    ) -> Self {
        let link_path = out_path.with_extension("o");
        let opt_path = out_path.with_extension("optimized.o");
        let sym_path = out_path.with_extension("symbols.txt");

        tracing::debug!(%target, ?cpu, ?feature, ?out_path, "new session created");

        Session {
            target,
            cpu,
            feature,
            symbols: Vec::new(),
            files: Vec::new(),
            link_path,
            opt_path,
            sym_path,
            out_path,
        }
    }

    /// Add a file, like an rlib or bitcode file that should be linked
    pub fn add_file(&mut self, path: PathBuf) {
        self.files.push(path);
    }

    /// Add a Vec of symbols to the list of exported symbols
    pub fn add_exported_symbols(&mut self, symbols: Vec<String>) {
        self.symbols.extend(symbols);
    }

    /// Reads every file that was added to the session and link them without optimization.
    ///
    /// The resulting artifact will be written to a file that can later be read to perform
    /// optimizations and/or compilation from bitcode to the final artifact.
    fn link(&mut self) -> anyhow::Result<()> {
        tracing::info!("Linking {} files using llvm-link", self.files.len());

        let llvm_link_output = std::process::Command::new("llvm-link")
            .arg("--ignore-non-bitcode")
            .args(&self.files)
            .arg("-o")
            .arg(&self.link_path)
            .output()
            .context("An error occurred when calling llvm-link. Make sure the llvm-tools component is installed.")?;

        if !llvm_link_output.status.success() {
            tracing::error!(
                "llvm-link returned with Exit status: {}\n stdout: {}\n stderr: {}",
                llvm_link_output.status,
                String::from_utf8(llvm_link_output.stdout).unwrap(),
                String::from_utf8(llvm_link_output.stderr).unwrap(),
            );
            anyhow::bail!("llvm-link failed to link files {:?}", self.files);
        }

        Ok(())
    }

    /// Optimize and compile to native format using `opt` and `llc`
    ///
    /// Before this can be called `link` needs to be called
    fn optimize(&mut self, optimization: Optimization, mut debug: bool) -> anyhow::Result<()> {
        let mut passes = format!("default<{}>", optimization);

        // Debug symbol generation is broken for ptx ISA versions older than 7.0.
        // See issue: https://github.com/rust-lang/rust/issues/99248
        if debug && self.target == crate::Target::Nvptx64NvidiaCuda {
            // The reason it's sufficient to check for `ptx70` is that
            // - ptx versions are controlled by target features in llvm
            // - In rust, newer ptx versions automatically actives older ones
            let explicit_sufficient_version =
                self.feature.as_ref().is_some_and(|feat| feat.contains("ptx70"));

            // When an SM version is specified, LLVM will choose a corresponding PTX ISA version.
            // The list below contains SM versions that LLVM maps to PTX ISA versions older than 7.0.
            // If LLVM updates its defaults, these entries can be removed.
            // Likewise, if debug info generation is fixed for PTX < 7.0,
            // any SM versions that start working can also be removed.
            // Note: newly introduced SM versions will always map to a PTX ISA ≥ 7.0,
            // so they do not need to be listed here.
            let implicit_sufficient_version = self.cpu.as_ref().is_some_and(|cpu| {
                ![
                    "sm_20", "sm_30", "sm_35", "sm_50", "sm_52", "sm_53", "sm_60", "sm_61",
                    "sm_62", "sm_70", "sm_72", "sm_75",
                ]
                .contains(&cpu.as_str())
            });

            if !explicit_sufficient_version && !implicit_sufficient_version {
                tracing::warn!("PTX version < 7.0 - stripping debug symbols");
                debug = false;
            }
        }

        // We add an internalize pass as the rust compiler as we require exported symbols to be explicitly marked
        passes.push_str(",internalize,globaldce");
        let symbol_file_content = self.symbols.iter().fold(String::new(), |s, x| s + &x + "\n");
        std::fs::write(&self.sym_path, symbol_file_content)
            .context(format!("Failed to write symbol file: {}", self.sym_path.display()))?;

        tracing::info!("optimizing bitcode with passes: {}", passes);
        let mut opt_cmd = std::process::Command::new("opt");
        opt_cmd
            .arg(&self.link_path)
            .arg("-o")
            .arg(&self.opt_path)
            .arg(format!("--internalize-public-api-file={}", self.sym_path.display()))
            .arg(format!("--passes={}", passes));

        if !debug {
            opt_cmd.arg("--strip-debug");
        }

        let opt_output = opt_cmd.output().context(
            "An error occurred when calling opt. Make sure the llvm-tools component is installed.",
        )?;

        if !opt_output.status.success() {
            tracing::error!(
                "opt returned with Exit status: {}\n stdout: {}\n stderr: {}",
                opt_output.status,
                String::from_utf8(opt_output.stdout).unwrap(),
                String::from_utf8(opt_output.stderr).unwrap(),
            );
            anyhow::bail!("opt failed optimize bitcode: {}", self.link_path.display());
        };

        Ok(())
    }

    /// Compile the optimized bitcode file to native format using `llc`
    ///
    /// Before this can be called `optimize` needs to be called
    fn compile(&mut self) -> anyhow::Result<()> {
        let mut lcc_command = std::process::Command::new("llc");

        if let Some(mcpu) = &self.cpu {
            lcc_command.arg("--mcpu").arg(mcpu);
        }

        if let Some(mattr) = &self.feature {
            lcc_command.arg(&format!("--mattr={}", mattr));
        }

        let lcc_output = lcc_command
            .arg(&self.opt_path)
            .arg("-o").arg(&self.out_path)
            .output()
            .context("An error occurred when calling llc. Make sure the llvm-tools component is installed.")?;

        if !lcc_output.status.success() {
            tracing::error!(
                "llc returned with Exit status: {}\n stdout: {}\n stderr: {}",
                lcc_output.status,
                String::from_utf8(lcc_output.stdout).unwrap(),
                String::from_utf8(lcc_output.stderr).unwrap(),
            );

            anyhow::bail!(
                "llc failed to compile {} into {}",
                self.opt_path.display(),
                self.out_path.display()
            );
        }

        Ok(())
    }

    /// Links, optimizes and compiles to the native format
    pub fn lto(&mut self, optimization: crate::Optimization, debug: bool) -> anyhow::Result<()> {
        self.link()?;
        self.optimize(optimization, debug)?;
        self.compile()
    }
}
