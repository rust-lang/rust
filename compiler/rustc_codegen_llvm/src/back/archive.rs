//! A helper class for dealing with static archives

use std::ffi::{CStr, CString};
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
    should_update_symbols: bool,
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
            should_update_symbols: false,
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

    /// Indicate that the next call to `build` should update all symbols in
    /// the archive (equivalent to running 'ar s' over it).
    fn update_symbols(&mut self) {
        self.should_update_symbols = true;
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

        // we've checked for \0 characters in the library name already
        let dll_name_z = CString::new(lib_name).unwrap();
        // All import names are Rust identifiers and therefore cannot contain \0 characters.
        // FIXME: when support for #[link_name] implemented, ensure that import.name values don't
        // have any \0 characters
        let import_name_vector: Vec<CString> = dll_imports
            .iter()
            .map(|import: &DllImport| {
                if self.config.sess.target.arch == "x86" {
                    LlvmArchiveBuilder::i686_decorated_name(import)
                } else {
                    CString::new(import.name.to_string()).unwrap()
                }
            })
            .collect();

        let output_path_z = rustc_fs_util::path_to_c_string(&output_path);

        tracing::trace!("invoking LLVMRustWriteImportLibrary");
        tracing::trace!("  dll_name {:#?}", dll_name_z);
        tracing::trace!("  output_path {}", output_path.display());
        tracing::trace!(
            "  import names: {}",
            dll_imports.iter().map(|import| import.name.to_string()).collect::<Vec<_>>().join(", "),
        );

        let ffi_exports: Vec<LLVMRustCOFFShortExport> = import_name_vector
            .iter()
            .map(|name_z| LLVMRustCOFFShortExport::from_name(name_z.as_ptr()))
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
        let should_update_symbols = self.should_update_symbols;

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
                should_update_symbols,
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

    fn i686_decorated_name(import: &DllImport) -> CString {
        let name = import.name;
        // We verified during construction that `name` does not contain any NULL characters, so the
        // conversion to CString is guaranteed to succeed.
        CString::new(match import.calling_convention {
            DllCallingConvention::C => format!("_{}", name),
            DllCallingConvention::Stdcall(arg_list_size) => format!("_{}@{}", name, arg_list_size),
            DllCallingConvention::Fastcall(arg_list_size) => format!("@{}@{}", name, arg_list_size),
            DllCallingConvention::Vectorcall(arg_list_size) => {
                format!("{}@@{}", name, arg_list_size)
            }
        })
        .unwrap()
    }
}

fn string_to_io_error(s: String) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("bad archive: {}", s))
}
