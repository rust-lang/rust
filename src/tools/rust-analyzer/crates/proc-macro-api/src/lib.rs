//! Client-side Proc-Macro crate
//!
//! We separate proc-macro expanding logic to an extern program to allow
//! different implementations (e.g. wasm or dylib loading). And this crate
//! is used to provide basic infrastructure for communication between two
//! processes: Client (RA itself), Server (the external program)

pub mod legacy_protocol;
mod process;

use paths::{AbsPath, AbsPathBuf};
use span::{ErasedFileAstId, FIXUP_ERASED_FILE_AST_ID_MARKER, Span};
use std::{fmt, io, sync::Arc, time::SystemTime};

use crate::process::ProcMacroServerProcess;

/// The versions of the server protocol
pub mod version {
    pub const NO_VERSION_CHECK_VERSION: u32 = 0;
    pub const VERSION_CHECK_VERSION: u32 = 1;
    pub const ENCODE_CLOSE_SPAN_VERSION: u32 = 2;
    pub const HAS_GLOBAL_SPANS: u32 = 3;
    pub const RUST_ANALYZER_SPAN_SUPPORT: u32 = 4;
    /// Whether literals encode their kind as an additional u32 field and idents their rawness as a u32 field.
    pub const EXTENDED_LEAF_DATA: u32 = 5;
    pub const HASHED_AST_ID: u32 = 6;

    /// Current API version of the proc-macro protocol.
    pub const CURRENT_API_VERSION: u32 = HASHED_AST_ID;
}

/// Represents different kinds of procedural macros that can be expanded by the external server.
#[derive(Copy, Clone, Eq, PartialEq, Debug, serde_derive::Serialize, serde_derive::Deserialize)]
pub enum ProcMacroKind {
    /// A macro that derives implementations for a struct or enum.
    CustomDerive,
    /// An attribute-like procedural macro.
    Attr,
    // This used to be called FuncLike, so that's what the server expects currently.
    #[serde(alias = "Bang")]
    #[serde(rename(serialize = "FuncLike", deserialize = "FuncLike"))]
    Bang,
}

/// A handle to an external process which load dylibs with macros (.so or .dll)
/// and runs actual macro expansion functions.
#[derive(Debug)]
pub struct ProcMacroClient {
    /// Currently, the proc macro process expands all procedural macros sequentially.
    ///
    /// That means that concurrent salsa requests may block each other when expanding proc macros,
    /// which is unfortunate, but simple and good enough for the time being.
    process: Arc<ProcMacroServerProcess>,
    path: AbsPathBuf,
}

/// Represents a dynamically loaded library containing procedural macros.
pub struct MacroDylib {
    path: AbsPathBuf,
}

impl MacroDylib {
    /// Creates a new MacroDylib instance with the given path.
    pub fn new(path: AbsPathBuf) -> MacroDylib {
        MacroDylib { path }
    }
}

/// A handle to a specific proc-macro (a `#[proc_macro]` annotated function).
///
/// It exists within the context of a specific proc-macro server -- currently
/// we share a single expander process for all macros within a workspace.
#[derive(Debug, Clone)]
pub struct ProcMacro {
    process: Arc<ProcMacroServerProcess>,
    dylib_path: Arc<AbsPathBuf>,
    name: Box<str>,
    kind: ProcMacroKind,
    dylib_last_modified: Option<SystemTime>,
}

impl Eq for ProcMacro {}
impl PartialEq for ProcMacro {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.kind == other.kind
            && self.dylib_path == other.dylib_path
            && self.dylib_last_modified == other.dylib_last_modified
            && Arc::ptr_eq(&self.process, &other.process)
    }
}

/// Represents errors encountered when communicating with the proc-macro server.
#[derive(Clone, Debug)]
pub struct ServerError {
    pub message: String,
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

impl ProcMacroClient {
    /// Spawns an external process as the proc macro server and returns a client connected to it.
    pub fn spawn<'a>(
        process_path: &AbsPath,
        env: impl IntoIterator<
            Item = (impl AsRef<std::ffi::OsStr>, &'a Option<impl 'a + AsRef<std::ffi::OsStr>>),
        > + Clone,
    ) -> io::Result<ProcMacroClient> {
        let process = ProcMacroServerProcess::run(process_path, env)?;
        Ok(ProcMacroClient { process: Arc::new(process), path: process_path.to_owned() })
    }

    /// Returns the absolute path to the proc-macro server.
    pub fn server_path(&self) -> &AbsPath {
        &self.path
    }

    /// Loads a proc-macro dylib into the server process returning a list of `ProcMacro`s loaded.
    pub fn load_dylib(&self, dylib: MacroDylib) -> Result<Vec<ProcMacro>, ServerError> {
        let _p = tracing::info_span!("ProcMacroServer::load_dylib").entered();
        let macros = self.process.find_proc_macros(&dylib.path)?;

        let dylib_path = Arc::new(dylib.path);
        let dylib_last_modified = std::fs::metadata(dylib_path.as_path())
            .ok()
            .and_then(|metadata| metadata.modified().ok());
        match macros {
            Ok(macros) => Ok(macros
                .into_iter()
                .map(|(name, kind)| ProcMacro {
                    process: self.process.clone(),
                    name: name.into(),
                    kind,
                    dylib_path: dylib_path.clone(),
                    dylib_last_modified,
                })
                .collect()),
            Err(message) => Err(ServerError { message, io: None }),
        }
    }

    /// Checks if the proc-macro server has exited.
    pub fn exited(&self) -> Option<&ServerError> {
        self.process.exited()
    }
}

impl ProcMacro {
    /// Returns the name of the procedural macro.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the type of procedural macro.
    pub fn kind(&self) -> ProcMacroKind {
        self.kind
    }

    fn needs_fixup_change(&self) -> bool {
        let version = self.process.version();
        (version::RUST_ANALYZER_SPAN_SUPPORT..version::HASHED_AST_ID).contains(&version)
    }

    /// On some server versions, the fixup ast id is different than ours. So change it to match.
    fn change_fixup_to_match_old_server(&self, tt: &mut tt::TopSubtree<Span>) {
        const OLD_FIXUP_AST_ID: ErasedFileAstId = ErasedFileAstId::from_raw(!0 - 1);
        let change_ast_id = |ast_id: &mut ErasedFileAstId| {
            if *ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
                *ast_id = OLD_FIXUP_AST_ID;
            } else if *ast_id == OLD_FIXUP_AST_ID {
                // Swap between them, that means no collision plus the change can be reversed by doing itself.
                *ast_id = FIXUP_ERASED_FILE_AST_ID_MARKER;
            }
        };

        for tt in &mut tt.0 {
            match tt {
                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident { span, .. }))
                | tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal { span, .. }))
                | tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { span, .. })) => {
                    change_ast_id(&mut span.anchor.ast_id);
                }
                tt::TokenTree::Subtree(subtree) => {
                    change_ast_id(&mut subtree.delimiter.open.anchor.ast_id);
                    change_ast_id(&mut subtree.delimiter.close.anchor.ast_id);
                }
            }
        }
    }

    /// Expands the procedural macro by sending an expansion request to the server.
    /// This includes span information and environmental context.
    pub fn expand(
        &self,
        subtree: tt::SubtreeView<'_, Span>,
        attr: Option<tt::SubtreeView<'_, Span>>,
        env: Vec<(String, String)>,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
        current_dir: String,
    ) -> Result<Result<tt::TopSubtree<Span>, String>, ServerError> {
        let (mut subtree, mut attr) = (subtree, attr);
        let (mut subtree_changed, mut attr_changed);
        if self.needs_fixup_change() {
            subtree_changed = tt::TopSubtree::from_subtree(subtree);
            self.change_fixup_to_match_old_server(&mut subtree_changed);
            subtree = subtree_changed.view();

            if let Some(attr) = &mut attr {
                attr_changed = tt::TopSubtree::from_subtree(*attr);
                self.change_fixup_to_match_old_server(&mut attr_changed);
                *attr = attr_changed.view();
            }
        }

        legacy_protocol::expand(
            self,
            subtree,
            attr,
            env,
            def_site,
            call_site,
            mixed_site,
            current_dir,
        )
    }
}
