use std::path::{Path, PathBuf};

use crate::command::Command;
use crate::env::env_var;

/// Construct a new `llvm-readobj` invocation with the `GNU` output style.
/// This assumes that `llvm-readobj` is available at `$LLVM_BIN_DIR/llvm-readobj`.
#[track_caller]
pub fn llvm_readobj() -> LlvmReadobj {
    LlvmReadobj::new()
}

/// Construct a new `llvm-profdata` invocation. This assumes that `llvm-profdata` is available
/// at `$LLVM_BIN_DIR/llvm-profdata`.
#[track_caller]
pub fn llvm_profdata() -> LlvmProfdata {
    LlvmProfdata::new()
}

/// Construct a new `llvm-filecheck` invocation. This assumes that `llvm-filecheck` is available
/// at `$LLVM_FILECHECK`.
#[track_caller]
pub fn llvm_filecheck() -> LlvmFilecheck {
    LlvmFilecheck::new()
}

/// Construct a new `llvm-objdump` invocation. This assumes that `llvm-objdump` is available
/// at `$LLVM_BIN_DIR/llvm-objdump`.
pub fn llvm_objdump() -> LlvmObjdump {
    LlvmObjdump::new()
}

/// Construct a new `llvm-ar` invocation. This assumes that `llvm-ar` is available
/// at `$LLVM_BIN_DIR/llvm-ar`.
pub fn llvm_ar() -> LlvmAr {
    LlvmAr::new()
}

/// Construct a new `llvm-nm` invocation. This assumes that `llvm-nm` is available
/// at `$LLVM_BIN_DIR/llvm-nm`.
pub fn llvm_nm() -> LlvmNm {
    LlvmNm::new()
}

/// Construct a new `llvm-bcanalyzer` invocation. This assumes that `llvm-bcanalyzer` is available
/// at `$LLVM_BIN_DIR/llvm-bcanalyzer`.
pub fn llvm_bcanalyzer() -> LlvmBcanalyzer {
    LlvmBcanalyzer::new()
}

/// Construct a new `llvm-dwarfdump` invocation. This assumes that `llvm-dwarfdump` is available
/// at `$LLVM_BIN_DIR/llvm-dwarfdump`.
pub fn llvm_dwarfdump() -> LlvmDwarfdump {
    LlvmDwarfdump::new()
}

/// Construct a new `llvm-pdbutil` invocation. This assumes that `llvm-pdbutil` is available
/// at `$LLVM_BIN_DIR/llvm-pdbutil`.
pub fn llvm_pdbutil() -> LlvmPdbutil {
    LlvmPdbutil::new()
}

/// Construct a new `llvm-as` invocation. This assumes that `llvm-as` is available
/// at `$LLVM_BIN_DIR/llvm-as`.
pub fn llvm_as() -> LlvmAs {
    LlvmAs::new()
}

/// Construct a new `llvm-dis` invocation. This assumes that `llvm-dis` is available
/// at `$LLVM_BIN_DIR/llvm-dis`.
pub fn llvm_dis() -> LlvmDis {
    LlvmDis::new()
}

/// Construct a new `llvm-objcopy` invocation. This assumes that `llvm-objcopy` is available
/// at `$LLVM_BIN_DIR/llvm-objcopy`.
pub fn llvm_objcopy() -> LlvmObjcopy {
    LlvmObjcopy::new()
}

/// A `llvm-readobj` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmReadobj {
    cmd: Command,
}

/// A `llvm-profdata` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmProfdata {
    cmd: Command,
}

/// A `llvm-filecheck` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmFilecheck {
    cmd: Command,
}

/// A `llvm-objdump` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmObjdump {
    cmd: Command,
}

/// A `llvm-ar` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmAr {
    cmd: Command,
}

/// A `llvm-nm` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmNm {
    cmd: Command,
}

/// A `llvm-bcanalyzer` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmBcanalyzer {
    cmd: Command,
}

/// A `llvm-dwarfdump` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmDwarfdump {
    cmd: Command,
}

/// A `llvm-pdbutil` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmPdbutil {
    cmd: Command,
}

/// A `llvm-as` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmAs {
    cmd: Command,
}

/// A `llvm-dis` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmDis {
    cmd: Command,
}

/// A `llvm-objcopy` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct LlvmObjcopy {
    cmd: Command,
}

crate::macros::impl_common_helpers!(LlvmReadobj);
crate::macros::impl_common_helpers!(LlvmProfdata);
crate::macros::impl_common_helpers!(LlvmFilecheck);
crate::macros::impl_common_helpers!(LlvmObjdump);
crate::macros::impl_common_helpers!(LlvmAr);
crate::macros::impl_common_helpers!(LlvmNm);
crate::macros::impl_common_helpers!(LlvmBcanalyzer);
crate::macros::impl_common_helpers!(LlvmDwarfdump);
crate::macros::impl_common_helpers!(LlvmPdbutil);
crate::macros::impl_common_helpers!(LlvmAs);
crate::macros::impl_common_helpers!(LlvmDis);
crate::macros::impl_common_helpers!(LlvmObjcopy);

/// Generate the path to the bin directory of LLVM.
#[must_use]
pub fn llvm_bin_dir() -> PathBuf {
    let llvm_bin_dir = env_var("LLVM_BIN_DIR");
    PathBuf::from(llvm_bin_dir)
}

impl LlvmReadobj {
    /// Construct a new `llvm-readobj` invocation with the `GNU` output style.
    /// This assumes that `llvm-readobj` is available at `$LLVM_BIN_DIR/llvm-readobj`.
    #[track_caller]
    pub fn new() -> Self {
        let llvm_readobj = llvm_bin_dir().join("llvm-readobj");
        let cmd = Command::new(llvm_readobj);
        let mut readobj = Self { cmd };
        readobj.elf_output_style("GNU");
        readobj
    }

    /// Specify the format of the ELF information.
    ///
    /// Valid options are `LLVM` (default), `GNU`, and `JSON`.
    pub fn elf_output_style(&mut self, style: &str) -> &mut Self {
        self.cmd.arg("--elf-output-style");
        self.cmd.arg(style);
        self
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Pass `--file-header` to display file headers.
    pub fn file_header(&mut self) -> &mut Self {
        self.cmd.arg("--file-header");
        self
    }

    /// Pass `--program-headers` to display program headers.
    pub fn program_headers(&mut self) -> &mut Self {
        self.cmd.arg("--program-headers");
        self
    }

    /// Pass `--symbols` to display the symbol table, including both local
    /// and global symbols.
    pub fn symbols(&mut self) -> &mut Self {
        self.cmd.arg("--symbols");
        self
    }

    /// Pass `--dynamic-table` to display the dynamic symbol table.
    pub fn dynamic_table(&mut self) -> &mut Self {
        self.cmd.arg("--dynamic-table");
        self
    }

    /// Specify the section to display.
    pub fn section(&mut self, section: &str) -> &mut Self {
        self.cmd.arg("--string-dump");
        self.cmd.arg(section);
        self
    }
}

impl LlvmProfdata {
    /// Construct a new `llvm-profdata` invocation. This assumes that `llvm-profdata` is available
    /// at `$LLVM_BIN_DIR/llvm-profdata`.
    #[track_caller]
    pub fn new() -> Self {
        let llvm_profdata = llvm_bin_dir().join("llvm-profdata");
        let cmd = Command::new(llvm_profdata);
        Self { cmd }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify the output file path.
    pub fn output<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("-o");
        self.cmd.arg(path.as_ref());
        self
    }

    /// Take several profile data files generated by PGO instrumentation and merge them
    /// together into a single indexed profile data file.
    pub fn merge(&mut self) -> &mut Self {
        self.cmd.arg("merge");
        self
    }
}

impl LlvmFilecheck {
    /// Construct a new `llvm-filecheck` invocation. This assumes that `llvm-filecheck` is available
    /// at `$LLVM_FILECHECK`.
    #[track_caller]
    pub fn new() -> Self {
        let llvm_filecheck = env_var("LLVM_FILECHECK");
        let cmd = Command::new(llvm_filecheck);
        Self { cmd }
    }

    /// Provide a buffer representing standard input containing patterns that will be matched
    /// against the `.patterns(path)` call.
    pub fn stdin_buf<I: AsRef<[u8]>>(&mut self, input: I) -> &mut Self {
        self.cmd.stdin_buf(input);
        self
    }

    /// Provide the patterns that need to be matched.
    pub fn patterns<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// `--input-file` option.
    pub fn input_file<P: AsRef<Path>>(&mut self, input_file: P) -> &mut Self {
        self.cmd.arg("--input-file");
        self.cmd.arg(input_file.as_ref());
        self
    }
}

impl LlvmObjdump {
    /// Construct a new `llvm-objdump` invocation. This assumes that `llvm-objdump` is available
    /// at `$LLVM_BIN_DIR/llvm-objdump`.
    pub fn new() -> Self {
        let llvm_objdump = llvm_bin_dir().join("llvm-objdump");
        let cmd = Command::new(llvm_objdump);
        Self { cmd }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Disassemble all executable sections found in the input files.
    pub fn disassemble(&mut self) -> &mut Self {
        self.cmd.arg("-d");
        self
    }
}

impl LlvmAr {
    /// Construct a new `llvm-ar` invocation. This assumes that `llvm-ar` is available
    /// at `$LLVM_BIN_DIR/llvm-ar`.
    pub fn new() -> Self {
        let llvm_ar = llvm_bin_dir().join("llvm-ar");
        let cmd = Command::new(llvm_ar);
        Self { cmd }
    }

    /// Automatically pass the commonly used arguments `rcus`, used for combining one or more
    /// input object files into one output static library file.
    pub fn obj_to_ar(&mut self) -> &mut Self {
        self.cmd.arg("rcus");
        self
    }

    /// Like `obj_to_ar` except creating a thin archive.
    pub fn obj_to_thin_ar(&mut self) -> &mut Self {
        self.cmd.arg("rcus").arg("--thin");
        self
    }

    /// Extract archive members back to files.
    pub fn extract(&mut self) -> &mut Self {
        self.cmd.arg("x");
        self
    }

    /// Print the table of contents.
    pub fn table_of_contents(&mut self) -> &mut Self {
        self.cmd.arg("t");
        self
    }

    /// Provide an output, then an input file. Bundled in one function, as llvm-ar has
    /// no "--output"-style flag.
    pub fn output_input(&mut self, out: impl AsRef<Path>, input: impl AsRef<Path>) -> &mut Self {
        self.cmd.arg(out.as_ref());
        self.cmd.arg(input.as_ref());
        self
    }
}

impl LlvmNm {
    /// Construct a new `llvm-nm` invocation. This assumes that `llvm-nm` is available
    /// at `$LLVM_BIN_DIR/llvm-nm`.
    pub fn new() -> Self {
        let llvm_nm = llvm_bin_dir().join("llvm-nm");
        let cmd = Command::new(llvm_nm);
        Self { cmd }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }
}

impl LlvmBcanalyzer {
    /// Construct a new `llvm-bcanalyzer` invocation. This assumes that `llvm-bcanalyzer` is available
    /// at `$LLVM_BIN_DIR/llvm-bcanalyzer`.
    pub fn new() -> Self {
        let llvm_bcanalyzer = llvm_bin_dir().join("llvm-bcanalyzer");
        let cmd = Command::new(llvm_bcanalyzer);
        Self { cmd }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }
}

impl LlvmDwarfdump {
    /// Construct a new `llvm-dwarfdump` invocation. This assumes that `llvm-dwarfdump` is available
    /// at `$LLVM_BIN_DIR/llvm-dwarfdump`.
    pub fn new() -> Self {
        let llvm_dwarfdump = llvm_bin_dir().join("llvm-dwarfdump");
        let cmd = Command::new(llvm_dwarfdump);
        Self { cmd }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }
}

impl LlvmPdbutil {
    /// Construct a new `llvm-pdbutil` invocation. This assumes that `llvm-pdbutil` is available
    /// at `$LLVM_BIN_DIR/llvm-pdbutil`.
    pub fn new() -> Self {
        let llvm_pdbutil = llvm_bin_dir().join("llvm-pdbutil");
        let cmd = Command::new(llvm_pdbutil);
        Self { cmd }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }
}

impl LlvmObjcopy {
    /// Construct a new `llvm-objcopy` invocation. This assumes that `llvm-objcopy` is available
    /// at `$LLVM_BIN_DIR/llvm-objcopy`.
    pub fn new() -> Self {
        let llvm_objcopy = llvm_bin_dir().join("llvm-objcopy");
        let cmd = Command::new(llvm_objcopy);
        Self { cmd }
    }

    /// Dump the contents of `section` into the file at `path`.
    #[track_caller]
    pub fn dump_section<S: AsRef<str>, P: AsRef<Path>>(
        &mut self,
        section_name: S,
        path: P,
    ) -> &mut Self {
        self.cmd.arg("--dump-section");
        self.cmd.arg(format!("{}={}", section_name.as_ref(), path.as_ref().to_str().unwrap()));
        self
    }
}

impl LlvmAs {
    /// Construct a new `llvm-as` invocation. This assumes that `llvm-as` is available
    /// at `$LLVM_BIN_DIR/llvm-as`.
    pub fn new() -> Self {
        let llvm_as = llvm_bin_dir().join("llvm-as");
        let cmd = Command::new(llvm_as);
        Self { cmd }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }
}

impl LlvmDis {
    /// Construct a new `llvm-dis` invocation. This assumes that `llvm-dis` is available
    /// at `$LLVM_BIN_DIR/llvm-dis`.
    pub fn new() -> Self {
        let llvm_dis = llvm_bin_dir().join("llvm-dis");
        let cmd = Command::new(llvm_dis);
        Self { cmd }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }
}
