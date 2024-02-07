//! Client-side Proc-Macro crate
//!
//! We separate proc-macro expanding logic to an extern program to allow
//! different implementations (e.g. wasm or dylib loading). And this crate
//! is used to provide basic infrastructure for communication between two
//! processes: Client (RA itself), Server (the external program)

#![warn(rust_2018_idioms, unused_lifetimes)]

pub mod msg;
mod process;
mod version;

use indexmap::IndexSet;
use paths::AbsPathBuf;
use span::Span;
use std::{
    fmt, io,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

use crate::{
    msg::{
        deserialize_span_data_index_map, flat::serialize_span_data_index_map, ExpandMacro,
        ExpnGlobals, FlatTree, PanicMessage, HAS_GLOBAL_SPANS, RUST_ANALYZER_SPAN_SUPPORT,
    },
    process::ProcMacroProcessSrv,
};

pub use version::{read_dylib_info, read_version, RustCInfo};

#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub enum ProcMacroKind {
    CustomDerive,
    FuncLike,
    Attr,
}

/// A handle to an external process which load dylibs with macros (.so or .dll)
/// and runs actual macro expansion functions.
#[derive(Debug)]
pub struct ProcMacroServer {
    /// Currently, the proc macro process expands all procedural macros sequentially.
    ///
    /// That means that concurrent salsa requests may block each other when expanding proc macros,
    /// which is unfortunate, but simple and good enough for the time being.
    ///
    /// Therefore, we just wrap the `ProcMacroProcessSrv` in a mutex here.
    process: Arc<Mutex<ProcMacroProcessSrv>>,
}

pub struct MacroDylib {
    path: AbsPathBuf,
}

impl MacroDylib {
    pub fn new(path: AbsPathBuf) -> MacroDylib {
        MacroDylib { path }
    }
}

/// A handle to a specific macro (a `#[proc_macro]` annotated function).
///
/// It exists within a context of a specific [`ProcMacroProcess`] -- currently
/// we share a single expander process for all macros.
#[derive(Debug, Clone)]
pub struct ProcMacro {
    process: Arc<Mutex<ProcMacroProcessSrv>>,
    dylib_path: AbsPathBuf,
    name: String,
    kind: ProcMacroKind,
}

impl Eq for ProcMacro {}
impl PartialEq for ProcMacro {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.kind == other.kind
            && self.dylib_path == other.dylib_path
            && Arc::ptr_eq(&self.process, &other.process)
    }
}

#[derive(Clone, Debug)]
pub struct ServerError {
    pub message: String,
    // io::Error isn't Clone for some reason
    pub io: Option<Arc<io::Error>>,
}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)?;
        if let Some(io) = &self.io {
            f.write_str(": ")?;
            io.fmt(f)?;
        }
        Ok(())
    }
}

pub struct MacroPanic {
    pub message: String,
}

impl ProcMacroServer {
    /// Spawns an external process as the proc macro server and returns a client connected to it.
    pub fn spawn(process_path: AbsPathBuf) -> io::Result<ProcMacroServer> {
        let process = ProcMacroProcessSrv::run(process_path)?;
        Ok(ProcMacroServer { process: Arc::new(Mutex::new(process)) })
    }

    pub fn load_dylib(&self, dylib: MacroDylib) -> Result<Vec<ProcMacro>, ServerError> {
        let _p = tracing::span!(tracing::Level::INFO, "ProcMacroClient::load_dylib").entered();
        let macros =
            self.process.lock().unwrap_or_else(|e| e.into_inner()).find_proc_macros(&dylib.path)?;

        match macros {
            Ok(macros) => Ok(macros
                .into_iter()
                .map(|(name, kind)| ProcMacro {
                    process: self.process.clone(),
                    name,
                    kind,
                    dylib_path: dylib.path.clone(),
                })
                .collect()),
            Err(message) => Err(ServerError { message, io: None }),
        }
    }
}

impl ProcMacro {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> ProcMacroKind {
        self.kind
    }

    pub fn expand(
        &self,
        subtree: &tt::Subtree<Span>,
        attr: Option<&tt::Subtree<Span>>,
        env: Vec<(String, String)>,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
    ) -> Result<Result<tt::Subtree<Span>, PanicMessage>, ServerError> {
        let version = self.process.lock().unwrap_or_else(|e| e.into_inner()).version();
        let current_dir = env
            .iter()
            .find(|(name, _)| name == "CARGO_MANIFEST_DIR")
            .map(|(_, value)| value.clone());

        let mut span_data_table = IndexSet::default();
        let def_site = span_data_table.insert_full(def_site).0;
        let call_site = span_data_table.insert_full(call_site).0;
        let mixed_site = span_data_table.insert_full(mixed_site).0;
        let task = ExpandMacro {
            macro_body: FlatTree::new(subtree, version, &mut span_data_table),
            macro_name: self.name.to_string(),
            attributes: attr.map(|subtree| FlatTree::new(subtree, version, &mut span_data_table)),
            lib: self.dylib_path.to_path_buf().into(),
            env,
            current_dir,
            has_global_spans: ExpnGlobals {
                serialize: version >= HAS_GLOBAL_SPANS,
                def_site,
                call_site,
                mixed_site,
            },
            span_data_table: if version >= RUST_ANALYZER_SPAN_SUPPORT {
                serialize_span_data_index_map(&span_data_table)
            } else {
                Vec::new()
            },
        };

        let response = self
            .process
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send_task(msg::Request::ExpandMacro(Box::new(task)))?;

        match response {
            msg::Response::ExpandMacro(it) => {
                Ok(it.map(|tree| FlatTree::to_subtree_resolved(tree, version, &span_data_table)))
            }
            msg::Response::ExpandMacroExtended(it) => Ok(it.map(|resp| {
                FlatTree::to_subtree_resolved(
                    resp.tree,
                    version,
                    &deserialize_span_data_index_map(&resp.span_data_table),
                )
            })),
            _ => Err(ServerError { message: "unexpected response".to_string(), io: None }),
        }
    }
}
