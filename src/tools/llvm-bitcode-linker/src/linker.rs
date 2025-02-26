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
            .context("An error occured when calling llvm-link. Make sure the llvm-tools component is installed.")?;

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

        // FIXME(@kjetilkjeka) Debug symbol generation is broken for nvptx64 so we must remove them even in debug mode
        if debug && self.target == crate::Target::Nvptx64NvidiaCuda {
            tracing::warn!("nvptx64 target detected - stripping debug symbols");
            debug = false;
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
            "An error occured when calling opt. Make sure the llvm-tools component is installed.",
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
            .context("An error occured when calling llc. Make sure the llvm-tools component is installed.")?;

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
