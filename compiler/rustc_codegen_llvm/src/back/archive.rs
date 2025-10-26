//! A helper class for dealing with static archives

use std::ffi::{CStr, c_char, c_void};
use std::io;

use rustc_codegen_ssa::back::archive::{
    ArArchiveBuilder, ArchiveBuilder, ArchiveBuilderBuilder, DEFAULT_OBJECT_READER, ObjectReader,
};
use rustc_session::Session;

use crate::llvm;

pub(crate) struct LlvmArchiveBuilderBuilder;

impl ArchiveBuilderBuilder for LlvmArchiveBuilderBuilder {
    fn new_archive_builder<'a>(&self, sess: &'a Session) -> Box<dyn ArchiveBuilder + 'a> {
        // Use the `object` crate to build archives, with a little bit of help from LLVM.
        Box::new(ArArchiveBuilder::new(sess, &LLVM_OBJECT_READER))
    }
}

// The object crate doesn't know how to get symbols for LLVM bitcode and COFF bigobj files.
// As such we need to use LLVM for them.

static LLVM_OBJECT_READER: ObjectReader = ObjectReader {
    get_symbols: get_llvm_object_symbols,
    is_64_bit_object_file: llvm_is_64_bit_object_file,
    is_ec_object_file: llvm_is_ec_object_file,
    is_any_arm64_coff: llvm_is_any_arm64_coff,
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
        let error = unsafe { *Box::from_raw(err as *mut io::Error) };
        // These are the magic constants for LLVM bitcode files:
        // https://github.com/llvm/llvm-project/blob/7eadc1960d199676f04add402bb0aa6f65b7b234/llvm/lib/BinaryFormat/Magic.cpp#L90-L97
        if buf.starts_with(&[0xDE, 0xCE, 0x17, 0x0B]) || buf.starts_with(&[b'B', b'C', 0xC0, 0xDE])
        {
            // For LLVM bitcode, failure to read the symbols is not fatal. The bitcode may have been
            // produced by a newer LLVM version that the one linked to rustc. This is fine provided
            // that the linker does use said newer LLVM version. We skip writing the symbols for the
            // bitcode to the symbol table of the archive. Traditional linkers don't like this, but
            // newer linkers like lld, mold and wild ignore the symbol table anyway, so if they link
            // against a new enough LLVM it will work out in the end.
            // LLVM's archive writer also has this same behavior of only warning about invalid
            // bitcode since https://github.com/llvm/llvm-project/pull/96848

            // We don't have access to the DiagCtxt here to produce a nice warning in the correct format.
            eprintln!("warning: Failed to read symbol table from LLVM bitcode: {}", error);
            return Ok(true);
        } else {
            return Err(error);
        }
    }

    unsafe extern "C" fn callback(state: *mut c_void, symbol_name: *const c_char) -> *mut c_void {
        let f = unsafe { &mut *(state as *mut &mut dyn FnMut(&[u8]) -> io::Result<()>) };
        match f(unsafe { CStr::from_ptr(symbol_name) }.to_bytes()) {
            Ok(()) => std::ptr::null_mut(),
            Err(err) => Box::into_raw(Box::new(err) as Box<io::Error>) as *mut c_void,
        }
    }

    unsafe extern "C" fn error_callback(error: *const c_char) -> *mut c_void {
        let error = unsafe { CStr::from_ptr(error) };
        Box::into_raw(Box::new(io::Error::new(
            io::ErrorKind::Other,
            format!("LLVM error: {}", error.to_string_lossy()),
        )) as Box<io::Error>) as *mut c_void
    }
}

fn llvm_is_64_bit_object_file(buf: &[u8]) -> bool {
    unsafe { llvm::LLVMRustIs64BitSymbolicFile(buf.as_ptr(), buf.len()) }
}

fn llvm_is_ec_object_file(buf: &[u8]) -> bool {
    unsafe { llvm::LLVMRustIsECObject(buf.as_ptr(), buf.len()) }
}

fn llvm_is_any_arm64_coff(buf: &[u8]) -> bool {
    unsafe { llvm::LLVMRustIsAnyArm64Coff(buf.as_ptr(), buf.len()) }
}
