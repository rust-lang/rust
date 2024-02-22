//! This module contains paths to types and functions Clippy needs to know
//! about.
//!
//! Whenever possible, please consider diagnostic items over hardcoded paths.
//! See <https://github.com/rust-lang/rust-clippy/issues/5393> for more information.

pub const APPLICABILITY: [&str; 2] = ["rustc_lint_defs", "Applicability"];
pub const APPLICABILITY_VALUES: [[&str; 3]; 4] = [
    ["rustc_lint_defs", "Applicability", "Unspecified"],
    ["rustc_lint_defs", "Applicability", "HasPlaceholders"],
    ["rustc_lint_defs", "Applicability", "MaybeIncorrect"],
    ["rustc_lint_defs", "Applicability", "MachineApplicable"],
];
pub const DIAGNOSTIC_BUILDER: [&str; 2] = ["rustc_errors", "DiagnosticBuilder"];
pub const BINARYHEAP_ITER: [&str; 5] = ["alloc", "collections", "binary_heap", "BinaryHeap", "iter"];
pub const BTREEMAP_CONTAINS_KEY: [&str; 6] = ["alloc", "collections", "btree", "map", "BTreeMap", "contains_key"];
pub const BTREEMAP_INSERT: [&str; 6] = ["alloc", "collections", "btree", "map", "BTreeMap", "insert"];
pub const BTREESET_ITER: [&str; 6] = ["alloc", "collections", "btree", "set", "BTreeSet", "iter"];
pub const CLONE_TRAIT_METHOD: [&str; 4] = ["core", "clone", "Clone", "clone"];
pub const CORE_ITER_CLONED: [&str; 6] = ["core", "iter", "traits", "iterator", "Iterator", "cloned"];
pub const CORE_ITER_COPIED: [&str; 6] = ["core", "iter", "traits", "iterator", "Iterator", "copied"];
pub const CORE_ITER_FILTER: [&str; 6] = ["core", "iter", "traits", "iterator", "Iterator", "filter"];
pub const CORE_RESULT_OK_METHOD: [&str; 4] = ["core", "result", "Result", "ok"];
pub const CSTRING_AS_C_STR: [&str; 5] = ["alloc", "ffi", "c_str", "CString", "as_c_str"];
pub const EARLY_CONTEXT: [&str; 2] = ["rustc_lint", "EarlyContext"];
pub const EARLY_LINT_PASS: [&str; 3] = ["rustc_lint", "passes", "EarlyLintPass"];
pub const F32_EPSILON: [&str; 4] = ["core", "f32", "<impl f32>", "EPSILON"];
pub const F64_EPSILON: [&str; 4] = ["core", "f64", "<impl f64>", "EPSILON"];
pub const FILE_OPTIONS: [&str; 4] = ["std", "fs", "File", "options"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const FUTURES_IO_ASYNCREADEXT: [&str; 3] = ["futures_util", "io", "AsyncReadExt"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const FUTURES_IO_ASYNCWRITEEXT: [&str; 3] = ["futures_util", "io", "AsyncWriteExt"];
pub const HASHMAP_CONTAINS_KEY: [&str; 6] = ["std", "collections", "hash", "map", "HashMap", "contains_key"];
pub const HASHMAP_INSERT: [&str; 6] = ["std", "collections", "hash", "map", "HashMap", "insert"];
pub const HASHMAP_ITER: [&str; 5] = ["std", "collections", "hash", "map", "Iter"];
pub const HASHMAP_ITER_MUT: [&str; 5] = ["std", "collections", "hash", "map", "IterMut"];
pub const HASHMAP_KEYS: [&str; 5] = ["std", "collections", "hash", "map", "Keys"];
pub const HASHMAP_VALUES: [&str; 5] = ["std", "collections", "hash", "map", "Values"];
pub const HASHMAP_DRAIN: [&str; 5] = ["std", "collections", "hash", "map", "Drain"];
pub const HASHMAP_VALUES_MUT: [&str; 5] = ["std", "collections", "hash", "map", "ValuesMut"];
pub const HASHSET_ITER_TY: [&str; 5] = ["std", "collections", "hash", "set", "Iter"];
pub const HASHSET_ITER: [&str; 6] = ["std", "collections", "hash", "set", "HashSet", "iter"];
pub const HASHSET_DRAIN: [&str; 5] = ["std", "collections", "hash", "set", "Drain"];
pub const IDENT: [&str; 3] = ["rustc_span", "symbol", "Ident"];
pub const IDENT_AS_STR: [&str; 4] = ["rustc_span", "symbol", "Ident", "as_str"];
pub const INSERT_STR: [&str; 4] = ["alloc", "string", "String", "insert_str"];
pub const ITERTOOLS_NEXT_TUPLE: [&str; 3] = ["itertools", "Itertools", "next_tuple"];
pub const KW_MODULE: [&str; 3] = ["rustc_span", "symbol", "kw"];
pub const LATE_CONTEXT: [&str; 2] = ["rustc_lint", "LateContext"];
pub const LATE_LINT_PASS: [&str; 3] = ["rustc_lint", "passes", "LateLintPass"];
pub const LINT: [&str; 2] = ["rustc_lint_defs", "Lint"];
pub const MSRV: [&str; 3] = ["clippy_config", "msrvs", "Msrv"];
pub const OPEN_OPTIONS_NEW: [&str; 4] = ["std", "fs", "OpenOptions", "new"];
pub const OS_STRING_AS_OS_STR: [&str; 5] = ["std", "ffi", "os_str", "OsString", "as_os_str"];
pub const OS_STR_TO_OS_STRING: [&str; 5] = ["std", "ffi", "os_str", "OsStr", "to_os_string"];
pub const PARKING_LOT_MUTEX_GUARD: [&str; 3] = ["lock_api", "mutex", "MutexGuard"];
pub const PARKING_LOT_RWLOCK_READ_GUARD: [&str; 3] = ["lock_api", "rwlock", "RwLockReadGuard"];
pub const PARKING_LOT_RWLOCK_WRITE_GUARD: [&str; 3] = ["lock_api", "rwlock", "RwLockWriteGuard"];
pub const PATH_BUF_AS_PATH: [&str; 4] = ["std", "path", "PathBuf", "as_path"];
pub const PATH_MAIN_SEPARATOR: [&str; 3] = ["std", "path", "MAIN_SEPARATOR"];
pub const PATH_TO_PATH_BUF: [&str; 4] = ["std", "path", "Path", "to_path_buf"];
#[cfg_attr(not(unix), allow(clippy::invalid_paths))]
pub const PERMISSIONS_FROM_MODE: [&str; 6] = ["std", "os", "unix", "fs", "PermissionsExt", "from_mode"];
pub const PUSH_STR: [&str; 4] = ["alloc", "string", "String", "push_str"];
pub const REGEX_BUILDER_NEW: [&str; 3] = ["regex", "RegexBuilder", "new"];
pub const REGEX_BYTES_BUILDER_NEW: [&str; 4] = ["regex", "bytes", "RegexBuilder", "new"];
pub const REGEX_BYTES_NEW: [&str; 4] = ["regex", "bytes", "Regex", "new"];
pub const REGEX_BYTES_SET_NEW: [&str; 4] = ["regex", "bytes", "RegexSet", "new"];
pub const REGEX_NEW: [&str; 3] = ["regex", "Regex", "new"];
pub const REGEX_SET_NEW: [&str; 3] = ["regex", "RegexSet", "new"];
pub const SERDE_DESERIALIZE: [&str; 3] = ["serde", "de", "Deserialize"];
pub const SERDE_DE_VISITOR: [&str; 3] = ["serde", "de", "Visitor"];
pub const SLICE_GET: [&str; 4] = ["core", "slice", "<impl [T]>", "get"];
pub const SLICE_INTO_VEC: [&str; 4] = ["alloc", "slice", "<impl [T]>", "into_vec"];
pub const SLICE_INTO: [&str; 4] = ["core", "slice", "<impl [T]>", "iter"];
pub const STD_IO_SEEK_FROM_CURRENT: [&str; 4] = ["std", "io", "SeekFrom", "Current"];
pub const STD_IO_SEEKFROM_START: [&str; 4] = ["std", "io", "SeekFrom", "Start"];
pub const STRING_AS_MUT_STR: [&str; 4] = ["alloc", "string", "String", "as_mut_str"];
pub const STRING_AS_STR: [&str; 4] = ["alloc", "string", "String", "as_str"];
pub const STRING_NEW: [&str; 4] = ["alloc", "string", "String", "new"];
pub const STR_BYTES: [&str; 4] = ["core", "str", "<impl str>", "bytes"];
pub const STR_CHARS: [&str; 4] = ["core", "str", "<impl str>", "chars"];
pub const STR_ENDS_WITH: [&str; 4] = ["core", "str", "<impl str>", "ends_with"];
pub const STR_LEN: [&str; 4] = ["core", "str", "<impl str>", "len"];
pub const STR_STARTS_WITH: [&str; 4] = ["core", "str", "<impl str>", "starts_with"];
pub const SYMBOL: [&str; 3] = ["rustc_span", "symbol", "Symbol"];
pub const SYMBOL_AS_STR: [&str; 4] = ["rustc_span", "symbol", "Symbol", "as_str"];
pub const SYMBOL_INTERN: [&str; 4] = ["rustc_span", "symbol", "Symbol", "intern"];
pub const SYMBOL_TO_IDENT_STRING: [&str; 4] = ["rustc_span", "symbol", "Symbol", "to_ident_string"];
pub const SYM_MODULE: [&str; 3] = ["rustc_span", "symbol", "sym"];
pub const SYNTAX_CONTEXT: [&str; 3] = ["rustc_span", "hygiene", "SyntaxContext"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const TOKIO_FILE_OPTIONS: [&str; 5] = ["tokio", "fs", "file", "File", "options"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const TOKIO_IO_ASYNCREADEXT: [&str; 5] = ["tokio", "io", "util", "async_read_ext", "AsyncReadExt"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const TOKIO_IO_ASYNCWRITEEXT: [&str; 5] = ["tokio", "io", "util", "async_write_ext", "AsyncWriteExt"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const TOKIO_IO_OPEN_OPTIONS: [&str; 4] = ["tokio", "fs", "open_options", "OpenOptions"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const TOKIO_IO_OPEN_OPTIONS_NEW: [&str; 5] = ["tokio", "fs", "open_options", "OpenOptions", "new"];
pub const VEC_AS_MUT_SLICE: [&str; 4] = ["alloc", "vec", "Vec", "as_mut_slice"];
pub const VEC_AS_SLICE: [&str; 4] = ["alloc", "vec", "Vec", "as_slice"];
pub const VEC_DEQUE_ITER: [&str; 5] = ["alloc", "collections", "vec_deque", "VecDeque", "iter"];
pub const VEC_FROM_ELEM: [&str; 3] = ["alloc", "vec", "from_elem"];
pub const VEC_NEW: [&str; 4] = ["alloc", "vec", "Vec", "new"];
pub const VEC_WITH_CAPACITY: [&str; 4] = ["alloc", "vec", "Vec", "with_capacity"];
pub const VEC_RESIZE: [&str; 4] = ["alloc", "vec", "Vec", "resize"];
pub const INSTANT_NOW: [&str; 4] = ["std", "time", "Instant", "now"];
pub const VEC_IS_EMPTY: [&str; 4] = ["alloc", "vec", "Vec", "is_empty"];
pub const VEC_POP: [&str; 4] = ["alloc", "vec", "Vec", "pop"];
pub const WAKER: [&str; 4] = ["core", "task", "wake", "Waker"];
pub const OPTION_UNWRAP: [&str; 4] = ["core", "option", "Option", "unwrap"];
pub const OPTION_EXPECT: [&str; 4] = ["core", "option", "Option", "expect"];
#[expect(clippy::invalid_paths)] // not sure why it thinks this, it works so
pub const BOOL_THEN: [&str; 4] = ["core", "bool", "<impl bool>", "then"];
pub const ALLOCATOR_GLOBAL: [&str; 3] = ["alloc", "alloc", "Global"];
