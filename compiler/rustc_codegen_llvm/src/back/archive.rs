//! A helper class for dealing with static archives

use std::env;
use std::ffi::{CStr, CString, OsString};
use std::io;
use std::mem;
use std::path::{Path, PathBuf};
use std::ptr;
use std::str;

use crate::llvm::archive_ro::{ArchiveRO, Child};
use crate::llvm::{self, ArchiveKind, LLVMMachineType, LLVMRustCOFFShortExport};
use rustc_codegen_ssa::back::archive::ArchiveBuilder;
use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_session::cstore::{DllCallingConvention, DllImport};
use rustc_session::Session;

struct ArchiveConfig<'a> {
    pub sess: &'a Session,
    pub dst: PathBuf,
    pub src: Option<PathBuf>,
}

/// Helper for adding many files to an archive.
#[must_use = "must call build() to finish building the archive"]
pub struct LlvmArchiveBuilder<'a> {
    config: ArchiveConfig<'a>,
    removals: Vec<String>,
    additions: Vec<Addition>,
    src_archive: Option<Option<ArchiveRO>>,
}

enum Addition {
    File { path: PathBuf, name_in_archive: String },
    Archive { path: PathBuf, archive: ArchiveRO, skip: Box<dyn FnMut(&str) -> bool> },
}

impl Addition {
    fn path(&self) -> &Path {
        match self {
            Addition::File { path, .. } | Addition::Archive { path, .. } => path,
        }
    }
}

fn is_relevant_child(c: &Child<'_>) -> bool {
    match c.name() {
        Some(name) => !name.contains("SYMDEF"),
        None => false,
    }
}

fn archive_config<'a>(sess: &'a Session, output: &Path, input: Option<&Path>) -> ArchiveConfig<'a> {
    ArchiveConfig { sess, dst: output.to_path_buf(), src: input.map(|p| p.to_path_buf()) }
}

/// Map machine type strings to values of LLVM's MachineTypes enum.
fn llvm_machine_type(cpu: &str) -> LLVMMachineType {
    match cpu {
        "x86_64" => LLVMMachineType::AMD64,
        "x86" => LLVMMachineType::I386,
        "aarch64" => LLVMMachineType::ARM64,
        "arm" => LLVMMachineType::ARM,
        _ => panic!("unsupported cpu type {}", cpu),
    }
}

impl<'a> ArchiveBuilder<'a> for LlvmArchiveBuilder<'a> {
    /// Creates a new static archive, ready for modifying the archive specified
    /// by `config`.
    fn new(sess: &'a Session, output: &Path, input: Option<&Path>) -> LlvmArchiveBuilder<'a> {
        let config = archive_config(sess, output, input);
        LlvmArchiveBuilder {
            config,
            removals: Vec::new(),
            additions: Vec::new(),
            src_archive: None,
        }
    }

    /// Removes a file from this archive
    fn remove_file(&mut self, file: &str) {
        self.removals.push(file.to_string());
    }

    /// Lists all files in an archive
    fn src_files(&mut self) -> Vec<String> {
        if self.src_archive().is_none() {
            return Vec::new();
        }

        let archive = self.src_archive.as_ref().unwrap().as_ref().unwrap();

        archive
            .iter()
            .filter_map(|child| child.ok())
            .filter(is_relevant_child)
            .filter_map(|child| child.name())
            .filter(|name| !self.removals.iter().any(|x| x == name))
            .map(|name| name.to_owned())
            .collect()
    }

    fn add_archive<F>(&mut self, archive: &Path, skip: F) -> io::Result<()>
    where
        F: FnMut(&str) -> bool + 'static,
    {
        let archive_ro = match ArchiveRO::open(archive) {
            Ok(ar) => ar,
            Err(e) => return Err(io::Error::new(io::ErrorKind::Other, e)),
        };
        if self.additions.iter().any(|ar| ar.path() == archive) {
            return Ok(());
        }
        self.additions.push(Addition::Archive {
            path: archive.to_path_buf(),
            archive: archive_ro,
            skip: Box::new(skip),
        });
        Ok(())
    }

    /// Adds an arbitrary file to this archive
    fn add_file(&mut self, file: &Path) {
        let name = file.file_name().unwrap().to_str().unwrap();
        self.additions
            .push(Addition::File { path: file.to_path_buf(), name_in_archive: name.to_owned() });
    }

    /// Combine the provided files, rlibs, and native libraries into a single
    /// `Archive`.
    fn build(mut self) {
        let kind = self.llvm_archive_kind().unwrap_or_else(|kind| {
            self.config.sess.fatal(&format!("Don't know how to build archive of type: {}", kind))
        });

        if let Err(e) = self.build_with_llvm(kind) {
            self.config.sess.fatal(&format!("failed to build archive: {}", e));
        }
    }

    fn inject_dll_import_lib(
        &mut self,
        lib_name: &str,
        dll_imports: &[DllImport],
        tmpdir: &MaybeTempDir,
    ) {
        let output_path = {
            let mut output_path: PathBuf = tmpdir.as_ref().to_path_buf();
            output_path.push(format!("{}_imports", lib_name));
            output_path.with_extension("lib")
        };

        let mingw_gnu_toolchain = self.config.sess.target.llvm_target.ends_with("pc-windows-gnu");

        let import_name_and_ordinal_vector: Vec<(String, Option<u16>)> = dll_imports
            .iter()
            .map(|import: &DllImport| {
                if self.config.sess.target.arch == "x86" {
                    (
                        LlvmArchiveBuilder::i686_decorated_name(import, mingw_gnu_toolchain),
                        import.ordinal,
                    )
                } else {
                    (import.name.to_string(), import.ordinal)
                }
            })
            .collect();

        if mingw_gnu_toolchain {
            // The binutils linker used on -windows-gnu targets cannot read the import
            // libraries generated by LLVM: in our attempts, the linker produced an .EXE
            // that loaded but crashed with an AV upon calling one of the imported
            // functions.  Therefore, use binutils to create the import library instead,
            // by writing a .DEF file to the temp dir and calling binutils's dlltool.
            let def_file_path =
                tmpdir.as_ref().join(format!("{}_imports", lib_name)).with_extension("def");

            let def_file_content = format!(
                "EXPORTS\n{}",
                import_name_and_ordinal_vector
                    .into_iter()
                    .map(|(name, ordinal)| {
                        match ordinal {
                            Some(n) => format!("{} @{} NONAME", name, n),
                            None => name,
                        }
                    })
                    .collect::<Vec<String>>()
                    .join("\n")
            );

            match std::fs::write(&def_file_path, def_file_content) {
                Ok(_) => {}
                Err(e) => {
                    self.config.sess.fatal(&format!("Error writing .DEF file: {}", e));
                }
            };

            let dlltool = find_binutils_dlltool(self.config.sess);
            let result = std::process::Command::new(dlltool)
                .args([
                    "-d",
                    def_file_path.to_str().unwrap(),
                    "-D",
                    lib_name,
                    "-l",
                    output_path.to_str().unwrap(),
                ])
                .output();

            match result {
                Err(e) => {
                    self.config.sess.fatal(&format!("Error calling dlltool: {}", e));
                }
                Ok(output) if !output.status.success() => self.config.sess.fatal(&format!(
                    "Dlltool could not create import library: {}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )),
                _ => {}
            }
        } else {
            // we've checked for \0 characters in the library name already
            let dll_name_z = CString::new(lib_name).unwrap();

            let output_path_z = rustc_fs_util::path_to_c_string(&output_path);

            tracing::trace!("invoking LLVMRustWriteImportLibrary");
            tracing::trace!("  dll_name {:#?}", dll_name_z);
            tracing::trace!("  output_path {}", output_path.display());
            tracing::trace!(
                "  import names: {}",
                dll_imports
                    .iter()
                    .map(|import| import.name.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            );

            // All import names are Rust identifiers and therefore cannot contain \0 characters.
            // FIXME: when support for #[link_name] is implemented, ensure that the import names
            // still don't contain any \0 characters.  Also need to check that the names don't
            // contain substrings like " @" or "NONAME" that are keywords or otherwise reserved
            // in definition files.
            let cstring_import_name_and_ordinal_vector: Vec<(CString, Option<u16>)> =
                import_name_and_ordinal_vector
                    .into_iter()
                    .map(|(name, ordinal)| (CString::new(name).unwrap(), ordinal))
                    .collect();

            let ffi_exports: Vec<LLVMRustCOFFShortExport> = cstring_import_name_and_ordinal_vector
                .iter()
                .map(|(name_z, ordinal)| LLVMRustCOFFShortExport::new(name_z.as_ptr(), *ordinal))
                .collect();
            let result = unsafe {
                crate::llvm::LLVMRustWriteImportLibrary(
                    dll_name_z.as_ptr(),
                    output_path_z.as_ptr(),
                    ffi_exports.as_ptr(),
                    ffi_exports.len(),
                    llvm_machine_type(&self.config.sess.target.arch) as u16,
                    !self.config.sess.target.is_like_msvc,
                )
            };

            if result == crate::llvm::LLVMRustResult::Failure {
                self.config.sess.fatal(&format!(
                    "Error creating import library for {}: {}",
                    lib_name,
                    llvm::last_error().unwrap_or("unknown LLVM error".to_string())
                ));
            }
        };

        self.add_archive(&output_path, |_| false).unwrap_or_else(|e| {
            self.config.sess.fatal(&format!(
                "failed to add native library {}: {}",
                output_path.display(),
                e
            ));
        });
    }
}

impl<'a> LlvmArchiveBuilder<'a> {
    fn src_archive(&mut self) -> Option<&ArchiveRO> {
        if let Some(ref a) = self.src_archive {
            return a.as_ref();
        }
        let src = self.config.src.as_ref()?;
        self.src_archive = Some(ArchiveRO::open(src).ok());
        self.src_archive.as_ref().unwrap().as_ref()
    }

    fn llvm_archive_kind(&self) -> Result<ArchiveKind, &str> {
        let kind = &*self.config.sess.target.archive_format;
        kind.parse().map_err(|_| kind)
    }

    fn build_with_llvm(&mut self, kind: ArchiveKind) -> io::Result<()> {
        let removals = mem::take(&mut self.removals);
        let mut additions = mem::take(&mut self.additions);
        let mut strings = Vec::new();
        let mut members = Vec::new();

        let dst = CString::new(self.config.dst.to_str().unwrap())?;

        unsafe {
            if let Some(archive) = self.src_archive() {
                for child in archive.iter() {
                    let child = child.map_err(string_to_io_error)?;
                    let child_name = match child.name() {
                        Some(s) => s,
                        None => continue,
                    };
                    if removals.iter().any(|r| r == child_name) {
                        continue;
                    }

                    let name = CString::new(child_name)?;
                    members.push(llvm::LLVMRustArchiveMemberNew(
                        ptr::null(),
                        name.as_ptr(),
                        Some(child.raw),
                    ));
                    strings.push(name);
                }
            }
            for addition in &mut additions {
                match addition {
                    Addition::File { path, name_in_archive } => {
                        let path = CString::new(path.to_str().unwrap())?;
                        let name = CString::new(name_in_archive.clone())?;
                        members.push(llvm::LLVMRustArchiveMemberNew(
                            path.as_ptr(),
                            name.as_ptr(),
                            None,
                        ));
                        strings.push(path);
                        strings.push(name);
                    }
                    Addition::Archive { archive, skip, .. } => {
                        for child in archive.iter() {
                            let child = child.map_err(string_to_io_error)?;
                            if !is_relevant_child(&child) {
                                continue;
                            }
                            let child_name = child.name().unwrap();
                            if skip(child_name) {
                                continue;
                            }

                            // It appears that LLVM's archive writer is a little
                            // buggy if the name we pass down isn't just the
                            // filename component, so chop that off here and
                            // pass it in.
                            //
                            // See LLVM bug 25877 for more info.
                            let child_name =
                                Path::new(child_name).file_name().unwrap().to_str().unwrap();
                            let name = CString::new(child_name)?;
                            let m = llvm::LLVMRustArchiveMemberNew(
                                ptr::null(),
                                name.as_ptr(),
                                Some(child.raw),
                            );
                            members.push(m);
                            strings.push(name);
                        }
                    }
                }
            }

            let r = llvm::LLVMRustWriteArchive(
                dst.as_ptr(),
                members.len() as libc::size_t,
                members.as_ptr() as *const &_,
                true,
                kind,
            );
            let ret = if r.into_result().is_err() {
                let err = llvm::LLVMRustGetLastError();
                let msg = if err.is_null() {
                    "failed to write archive".into()
                } else {
                    String::from_utf8_lossy(CStr::from_ptr(err).to_bytes())
                };
                Err(io::Error::new(io::ErrorKind::Other, msg))
            } else {
                Ok(())
            };
            for member in members {
                llvm::LLVMRustArchiveMemberFree(member);
            }
            ret
        }
    }

    fn i686_decorated_name(import: &DllImport, mingw: bool) -> String {
        let name = import.name;
        let prefix = if mingw { "" } else { "_" };

        match import.calling_convention {
            DllCallingConvention::C => format!("{}{}", prefix, name),
            DllCallingConvention::Stdcall(arg_list_size) => {
                format!("{}{}@{}", prefix, name, arg_list_size)
            }
            DllCallingConvention::Fastcall(arg_list_size) => format!("@{}@{}", name, arg_list_size),
            DllCallingConvention::Vectorcall(arg_list_size) => {
                format!("{}@@{}", name, arg_list_size)
            }
        }
    }
}

fn string_to_io_error(s: String) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("bad archive: {}", s))
}

fn find_binutils_dlltool(sess: &Session) -> OsString {
    assert!(sess.target.options.is_like_windows && !sess.target.options.is_like_msvc);
    if let Some(dlltool_path) = &sess.opts.debugging_opts.dlltool {
        return dlltool_path.clone().into_os_string();
    }

    let mut tool_name: OsString = if sess.host.arch != sess.target.arch {
        // We are cross-compiling, so we need the tool with the prefix matching our target
        if sess.target.arch == "x86" {
            "i686-w64-mingw32-dlltool"
        } else {
            "x86_64-w64-mingw32-dlltool"
        }
    } else {
        // We are not cross-compiling, so we just want `dlltool`
        "dlltool"
    }
    .into();

    if sess.host.options.is_like_windows {
        // If we're compiling on Windows, add the .exe suffix
        tool_name.push(".exe");
    }

    // NOTE: it's not clear how useful it is to explicitly search PATH.
    for dir in env::split_paths(&env::var_os("PATH").unwrap_or_default()) {
        let full_path = dir.join(&tool_name);
        if full_path.is_file() {
            return full_path.into_os_string();
        }
    }

    // The user didn't specify the location of the dlltool binary, and we weren't able
    // to find the appropriate one on the PATH.  Just return the name of the tool
    // and let the invocation fail with a hopefully useful error message.
    tool_name
}
