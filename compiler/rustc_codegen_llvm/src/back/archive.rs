//! A helper class for dealing with static archives

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::{io, mem, ptr, str};

use rustc_codegen_ssa::back::archive::{
    ArArchiveBuilder, ArchiveBuildFailure, ArchiveBuilder, ArchiveBuilderBuilder,
    DEFAULT_OBJECT_READER, ObjectReader, UnknownArchiveKind, try_extract_macho_fat_archive,
};
use rustc_session::Session;

use crate::llvm::archive_ro::{ArchiveRO, Child};
use crate::llvm::{self, ArchiveKind};

/// Helper for adding many files to an archive.
#[must_use = "must call build() to finish building the archive"]
pub(crate) struct LlvmArchiveBuilder<'a> {
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

impl<'a> ArchiveBuilder for LlvmArchiveBuilder<'a> {
    fn add_archive(
        &mut self,
        archive: &Path,
        skip: Box<dyn FnMut(&str) -> bool + 'static>,
    ) -> io::Result<()> {
        let mut archive = archive.to_path_buf();
        if self.sess.target.llvm_target.contains("-apple-macosx") {
            if let Some(new_archive) = try_extract_macho_fat_archive(self.sess, &archive)? {
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
            Err(error) => {
                self.sess.dcx().emit_fatal(ArchiveBuildFailure { path: output.to_owned(), error })
            }
        }
    }
}

pub(crate) struct LlvmArchiveBuilderBuilder;

impl ArchiveBuilderBuilder for LlvmArchiveBuilderBuilder {
    fn new_archive_builder<'a>(&self, sess: &'a Session) -> Box<dyn ArchiveBuilder + 'a> {
        // Keeping LlvmArchiveBuilder around in case of a regression caused by using
        // ArArchiveBuilder.
        // FIXME(#128955) remove a couple of months after #128936 gets merged in case
        // no regression is found.
        if false {
            Box::new(LlvmArchiveBuilder { sess, additions: Vec::new() })
        } else {
            Box::new(ArArchiveBuilder::new(sess, &LLVM_OBJECT_READER))
        }
    }
}

// The object crate doesn't know how to get symbols for LLVM bitcode and COFF bigobj files.
// As such we need to use LLVM for them.

static LLVM_OBJECT_READER: ObjectReader = ObjectReader {
    get_symbols: get_llvm_object_symbols,
    is_64_bit_object_file: llvm_is_64_bit_object_file,
    is_ec_object_file: llvm_is_ec_object_file,
    get_xcoff_member_alignment: DEFAULT_OBJECT_READER.get_xcoff_member_alignment,
};

#[deny(unsafe_op_in_unsafe_fn)]
fn get_llvm_object_symbols(
    buf: &[u8],
    f: &mut dyn FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<bool> {
    let mut state = Box::new(f);

    let err = unsafe {
        llvm::LLVMRustGetSymbols(
            buf.as_ptr(),
            buf.len(),
            (&raw mut *state) as *mut c_void,
            callback,
            error_callback,
        )
    };

    if err.is_null() {
        return Ok(true);
    } else {
        return Err(unsafe { *Box::from_raw(err as *mut io::Error) });
    }

    unsafe extern "C" fn callback(state: *mut c_void, symbol_name: *const c_char) -> *mut c_void {
        let f = unsafe { &mut *(state as *mut &mut dyn FnMut(&[u8]) -> io::Result<()>) };
        match f(unsafe { CStr::from_ptr(symbol_name) }.to_bytes()) {
            Ok(()) => std::ptr::null_mut(),
            Err(err) => Box::into_raw(Box::new(err)) as *mut c_void,
        }
    }

    unsafe extern "C" fn error_callback(error: *const c_char) -> *mut c_void {
        let error = unsafe { CStr::from_ptr(error) };
        Box::into_raw(Box::new(io::Error::new(
            io::ErrorKind::Other,
            format!("LLVM error: {}", error.to_string_lossy()),
        ))) as *mut c_void
    }
}

fn llvm_is_64_bit_object_file(buf: &[u8]) -> bool {
    unsafe { llvm::LLVMRustIs64BitSymbolicFile(buf.as_ptr(), buf.len()) }
}

fn llvm_is_ec_object_file(buf: &[u8]) -> bool {
    unsafe { llvm::LLVMRustIsECObject(buf.as_ptr(), buf.len()) }
}

impl<'a> LlvmArchiveBuilder<'a> {
    fn build_with_llvm(&mut self, output: &Path) -> io::Result<bool> {
        let kind = &*self.sess.target.archive_format;
        let kind = kind
            .parse::<ArchiveKind>()
            .map_err(|_| kind)
            .unwrap_or_else(|kind| self.sess.dcx().emit_fatal(UnknownArchiveKind { kind }));

        let mut additions = mem::take(&mut self.additions);
        let mut strings = Vec::new();
        let mut members = Vec::new();

        let dst = CString::new(output.to_str().unwrap())?;

        unsafe {
            for addition in &mut additions {
                match addition {
                    Addition::File { path, name_in_archive } => {
                        let path = CString::new(path.to_str().unwrap())?;
                        let name = CString::new(name_in_archive.as_bytes())?;
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
                self.sess.target.arch == "arm64ec",
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
    io::Error::new(io::ErrorKind::Other, format!("bad archive: {s}"))
}
