//! A helper class for dealing with static archives

use std::env;
use std::ffi::{CStr, CString, OsString};
use std::fs;
use std::io::{self, Write};
use std::mem;
use std::path::{Path, PathBuf};
use std::ptr;
use std::str;

use object::read::macho::FatArch;

use crate::common;
use crate::llvm::archive_ro::{ArchiveRO, Child};
use crate::llvm::{self, ArchiveKind, LLVMMachineType, LLVMRustCOFFShortExport};
use rustc_codegen_ssa::back::archive::{ArchiveBuilder, ArchiveBuilderBuilder};
use rustc_data_structures::memmap::Mmap;
use rustc_session::cstore::DllImport;
use rustc_session::Session;

/// Helper for adding many files to an archive.
#[must_use = "must call build() to finish building the archive"]
pub struct LlvmArchiveBuilder<'a> {
    sess: &'a Session,
    additions: Vec<Addition>,
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

fn try_filter_fat_archs(
    archs: object::read::Result<&[impl FatArch]>,
    target_arch: object::Architecture,
    archive_path: &Path,
    archive_map_data: &[u8],
) -> io::Result<Option<PathBuf>> {
    let archs = archs.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    let desired = match archs.iter().filter(|a| a.architecture() == target_arch).next() {
        Some(a) => a,
        None => return Ok(None),
    };

    let (mut new_f, extracted_path) = tempfile::Builder::new()
        .suffix(archive_path.file_name().unwrap())
        .tempfile()?
        .keep()
        .unwrap();

    new_f.write_all(
        desired.data(archive_map_data).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?,
    )?;

    Ok(Some(extracted_path))
}

fn try_extract_macho_fat_archive(
    sess: &Session,
    archive_path: &Path,
) -> io::Result<Option<PathBuf>> {
    let archive_map = unsafe { Mmap::map(fs::File::open(&archive_path)?)? };
    let target_arch = match sess.target.arch.as_ref() {
        "aarch64" => object::Architecture::Aarch64,
        "x86_64" => object::Architecture::X86_64,
        _ => return Ok(None),
    };

    match object::macho::FatHeader::parse(&*archive_map) {
        Ok(h) if h.magic.get(object::endian::BigEndian) == object::macho::FAT_MAGIC => {
            let archs = object::macho::FatHeader::parse_arch32(&*archive_map);
            try_filter_fat_archs(archs, target_arch, archive_path, &*archive_map)
        }
        Ok(h) if h.magic.get(object::endian::BigEndian) == object::macho::FAT_MAGIC_64 => {
            let archs = object::macho::FatHeader::parse_arch64(&*archive_map);
            try_filter_fat_archs(archs, target_arch, archive_path, &*archive_map)
        }
        // Not a FatHeader at all, just return None.
        _ => Ok(None),
    }
}

impl<'a> ArchiveBuilder<'a> for LlvmArchiveBuilder<'a> {
    fn add_archive(
        &mut self,
        archive: &Path,
        skip: Box<dyn FnMut(&str) -> bool + 'static>,
    ) -> io::Result<()> {
        let mut archive = archive.to_path_buf();
        if self.sess.target.llvm_target.contains("-apple-macosx") {
            if let Some(new_archive) = try_extract_macho_fat_archive(&self.sess, &archive)? {
                archive = new_archive
            }
        }
        let archive_ro = match ArchiveRO::open(&archive) {
            Ok(ar) => ar,
            Err(e) => return Err(io::Error::new(io::ErrorKind::Other, e)),
        };
        if self.additions.iter().any(|ar| ar.path() == archive) {
            return Ok(());
        }
        self.additions.push(Addition::Archive {
            path: archive,
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
    fn build(mut self: Box<Self>, output: &Path) -> bool {
        match self.build_with_llvm(output) {
            Ok(any_members) => any_members,
            Err(e) => self.sess.fatal(&format!("failed to build archive: {}", e)),
        }
    }
}

pub struct LlvmArchiveBuilderBuilder;

impl ArchiveBuilderBuilder for LlvmArchiveBuilderBuilder {
    fn new_archive_builder<'a>(&self, sess: &'a Session) -> Box<dyn ArchiveBuilder<'a> + 'a> {
        Box::new(LlvmArchiveBuilder { sess, additions: Vec::new() })
    }

    fn create_dll_import_lib(
        &self,
        sess: &Session,
        lib_name: &str,
        dll_imports: &[DllImport],
        tmpdir: &Path,
        is_direct_dependency: bool,
    ) -> PathBuf {
        let name_suffix = if is_direct_dependency { "_imports" } else { "_imports_indirect" };
        let output_path = {
            let mut output_path: PathBuf = tmpdir.to_path_buf();
            output_path.push(format!("{}{}", lib_name, name_suffix));
            output_path.with_extension("lib")
        };

        let target = &sess.target;
        let mingw_gnu_toolchain = common::is_mingw_gnu_toolchain(target);

        let import_name_and_ordinal_vector: Vec<(String, Option<u16>)> = dll_imports
            .iter()
            .map(|import: &DllImport| {
                if sess.target.arch == "x86" {
                    (
                        common::i686_decorated_name(import, mingw_gnu_toolchain, false),
                        import.ordinal(),
                    )
                } else {
                    (import.name.to_string(), import.ordinal())
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
                tmpdir.join(format!("{}{}", lib_name, name_suffix)).with_extension("def");

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
                    sess.fatal(&format!("Error writing .DEF file: {}", e));
                }
            };

            // --no-leading-underscore: For the `import_name_type` feature to work, we need to be
            // able to control the *exact* spelling of each of the symbols that are being imported:
            // hence we don't want `dlltool` adding leading underscores automatically.
            let dlltool = find_binutils_dlltool(sess);
            let result = std::process::Command::new(dlltool)
                .args([
                    "-d",
                    def_file_path.to_str().unwrap(),
                    "-D",
                    lib_name,
                    "-l",
                    output_path.to_str().unwrap(),
                    "--no-leading-underscore",
                ])
                .output();

            match result {
                Err(e) => {
                    sess.fatal(&format!("Error calling dlltool: {}", e));
                }
                Ok(output) if !output.status.success() => sess.fatal(&format!(
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

            trace!("invoking LLVMRustWriteImportLibrary");
            trace!("  dll_name {:#?}", dll_name_z);
            trace!("  output_path {}", output_path.display());
            trace!(
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
                    llvm_machine_type(&sess.target.arch) as u16,
                    !sess.target.is_like_msvc,
                )
            };

            if result == crate::llvm::LLVMRustResult::Failure {
                sess.fatal(&format!(
                    "Error creating import library for {}: {}",
                    lib_name,
                    llvm::last_error().unwrap_or("unknown LLVM error".to_string())
                ));
            }
        };

        output_path
    }
}

impl<'a> LlvmArchiveBuilder<'a> {
    fn build_with_llvm(&mut self, output: &Path) -> io::Result<bool> {
        let kind = &*self.sess.target.archive_format;
        let kind = kind.parse::<ArchiveKind>().map_err(|_| kind).unwrap_or_else(|kind| {
            self.sess.fatal(&format!("Don't know how to build archive of type: {}", kind))
        });

        let mut additions = mem::take(&mut self.additions);
        let mut strings = Vec::new();
        let mut members = Vec::new();

        let dst = CString::new(output.to_str().unwrap())?;

        unsafe {
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
                Ok(!members.is_empty())
            };
            for member in members {
                llvm::LLVMRustArchiveMemberFree(member);
            }
            ret
        }
    }
}

fn string_to_io_error(s: String) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("bad archive: {}", s))
}

fn find_binutils_dlltool(sess: &Session) -> OsString {
    assert!(sess.target.options.is_like_windows && !sess.target.options.is_like_msvc);
    if let Some(dlltool_path) = &sess.opts.unstable_opts.dlltool {
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
