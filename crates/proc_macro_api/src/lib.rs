//! Client-side Proc-Macro crate
//!
//! We separate proc-macro expanding logic to an extern program to allow
//! different implementations (e.g. wasm or dylib loading). And this crate
//! is used to provide basic infrastructure for communication between two
//! processes: Client (RA itself), Server (the external program)

pub mod msg;
mod process;
mod rpc;

use base_db::{Env, ProcMacro};
use std::{
    ffi::OsStr,
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

use tt::{SmolStr, Subtree};

use crate::process::{ProcMacroProcessSrv, ProcMacroProcessThread};

pub use rpc::{ExpansionResult, ExpansionTask, ListMacrosResult, ListMacrosTask, ProcMacroKind};

use memmap::Mmap;
use object::read::{File as BinaryFile, Object, ObjectSection};
use snap::read::FrameDecoder as SnapDecoder;

#[derive(Debug, Clone)]
struct ProcMacroProcessExpander {
    process: Arc<ProcMacroProcessSrv>,
    dylib_path: PathBuf,
    name: SmolStr,
}

impl Eq for ProcMacroProcessExpander {}
impl PartialEq for ProcMacroProcessExpander {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.dylib_path == other.dylib_path
            && Arc::ptr_eq(&self.process, &other.process)
    }
}

impl base_db::ProcMacroExpander for ProcMacroProcessExpander {
    fn expand(
        &self,
        subtree: &Subtree,
        attr: Option<&Subtree>,
        env: &Env,
    ) -> Result<Subtree, tt::ExpansionError> {
        let task = ExpansionTask {
            macro_body: subtree.clone(),
            macro_name: self.name.to_string(),
            attributes: attr.cloned(),
            lib: self.dylib_path.to_path_buf(),
            env: env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        };

        let result: ExpansionResult = self.process.send_task(msg::Request::ExpansionMacro(task))?;
        Ok(result.expansion)
    }
}

#[derive(Debug)]
pub struct ProcMacroClient {
    process: Arc<ProcMacroProcessSrv>,
    thread: ProcMacroProcessThread,
}

impl ProcMacroClient {
    pub fn extern_process(
        process_path: PathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> io::Result<ProcMacroClient> {
        let (thread, process) = ProcMacroProcessSrv::run(process_path, args)?;
        Ok(ProcMacroClient {
            process: Arc::new(process),
            thread,
        })
    }

    pub fn by_dylib_path(&self, dylib_path: &Path) -> Vec<ProcMacro> {
        let macros = match self.process.find_proc_macros(dylib_path) {
            Err(err) => {
                eprintln!("Failed to find proc macros. Error: {:#?}", err);
                return vec![];
            }
            Ok(macros) => macros,
        };

        macros
            .into_iter()
            .map(|(name, kind)| {
                let name = SmolStr::new(&name);
                let kind = match kind {
                    ProcMacroKind::CustomDerive => base_db::ProcMacroKind::CustomDerive,
                    ProcMacroKind::FuncLike => base_db::ProcMacroKind::FuncLike,
                    ProcMacroKind::Attr => base_db::ProcMacroKind::Attr,
                };
                let expander = Arc::new(ProcMacroProcessExpander {
                    process: self.process.clone(),
                    name: name.clone(),
                    dylib_path: dylib_path.into(),
                });

                ProcMacro {
                    name,
                    kind,
                    expander,
                }
            })
            .collect()
    }

    // This is used inside self.read_version() to locate the ".rustc" section
    // from a proc macro crate's binary file.
    fn read_section<'a>(&self, dylib_binary: &'a [u8], section_name: &str) -> io::Result<&'a [u8]> {
        BinaryFile::parse(dylib_binary)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
            .section_by_name(section_name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "section read error"))?
            .data()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    // Check the version of rustc that was used to compile a proc macro crate's
    // binary file.
    // A proc macro crate binary's ".rustc" section has following byte layout:
    // * [b'r',b'u',b's',b't',0,0,0,5] is the first 8 bytes
    // * ff060000 734e6150 is followed, it's the snappy format magic bytes,
    //   means bytes from here(including this sequence) are compressed in
    //   snappy compression format. Version info is inside here, so decompress
    //   this.
    // The bytes you get after decompressing the snappy format portion has
    // following layout:
    // * [b'r',b'u',b's',b't',0,0,0,5] is the first 8 bytes(again)
    // * [crate root bytes] next 4 bytes is to store crate root position,
    //   according to rustc's source code comment
    // * [length byte] next 1 byte tells us how many bytes we should read next
    //   for the version string's utf8 bytes
    // * [version string bytes encoded in utf8] <- GET THIS BOI
    // * [some more bytes that we don really care but still there] :-)
    // Check this issue for more about the bytes layout:
    // https://github.com/rust-analyzer/rust-analyzer/issues/6174
    #[allow(unused)]
    fn read_version(&self, dylib_path: &Path) -> io::Result<String> {
        let dylib_file = File::open(dylib_path)?;
        let dylib_mmaped = unsafe { Mmap::map(&dylib_file) }?;

        let dot_rustc = self.read_section(&dylib_mmaped, ".rustc")?;

        let header = &dot_rustc[..8];
        const EXPECTED_HEADER: [u8; 8] = [b'r', b'u', b's', b't', 0, 0, 0, 5];
        // check if header is valid
        if !(header == EXPECTED_HEADER) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!(".rustc section should start with header {:?}; header {:?} is actually presented.",EXPECTED_HEADER ,header)));
        }

        let snappy_portion = &dot_rustc[8..];

        let mut snappy_decoder = SnapDecoder::new(snappy_portion);

        // the bytes before version string bytes, so this basically is:
        // 8 bytes for [b'r',b'u',b's',b't',0,0,0,5]
        // 4 bytes for [crate root bytes]
        // 1 byte for length of version string
        // so 13 bytes in total, and we should check the 13th byte
        // to know the length
        let mut bytes_before_version = [0u8; 13];
        snappy_decoder.read_exact(&mut bytes_before_version)?;
        let length = bytes_before_version[12]; // what? can't use -1 indexing?

        let mut version_string_utf8 = vec![0u8; length as usize];
        snappy_decoder.read_exact(&mut version_string_utf8)?;
        let version_string = String::from_utf8(version_string_utf8);
        version_string.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}
