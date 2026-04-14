//! Tree Provider Protocol
//!
//! Defines a generic port-based RPC protocol for tree-structured data providers.
//! This unifies filesystem trees, XML/HTML DOMs, and other hierarchical data sources.
//!
//! Use cases:
//! - ISO9660 filesystem navigation
//! - XML/HTML document traversal
//! - VFS graph exploration
//! - Any hierarchical data structure

/// Node kind enumeration
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Directory or container node
    Directory = 0,
    /// File or leaf node with data
    File = 1,
    /// Symbolic link
    Symlink = 2,
    /// Other/unknown node type
    Other = 3,
}

/// Request types for tree provider RPC
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeProviderRequest {
    /// Get the root node ID
    Root = 0,
    /// List children of a node
    List = 1,
    /// Read data from a file node
    Read = 2,
    /// Get metadata for a node
    GetMetadata = 3,
}

/// Response types for tree provider RPC
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeProviderResponse {
    /// Success response
    Ok = 0,
    /// Error response
    Error = 1,
}

/// Error codes for tree provider operations
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeProviderError {
    /// Invalid parameters
    InvalidParam = 1,
    /// Node not found
    NotFound = 2,
    /// Operation not supported
    NotSupported = 3,
    /// I/O error
    IoError = 4,
    /// Node is not a directory (for List)
    NotADirectory = 5,
    /// Node is not a file (for Read)
    NotAFile = 6,
    /// Offset out of range
    OutOfRange = 7,
}

/// Wire format for Root request (empty)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RootRequest {
    // Empty - just the request type
}

/// Wire format for Root response
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RootResponse {
    /// Root node ID (opaque handle within provider)
    pub node_id: u64,
}

/// Wire format for List request
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ListRequest {
    /// Node ID to list children of
    pub node_id: u64,
}

/// Wire format for a single child entry in List response
/// The List response is: ListResponseHeader followed by N * ChildEntry
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChildEntry {
    /// Child node ID
    pub node_id: u64,
    /// Node kind (Directory, File, etc.)
    pub kind: u8,
    /// Name length in bytes
    pub name_len: u8,
    /// File size (0 for directories)
    pub size: u64,
    /// Reserved for alignment
    pub _reserved: [u8; 6],
    // Followed by name_len bytes of UTF-8 name data
}

/// Wire format for List response header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ListResponseHeader {
    /// Number of children
    pub count: u32,
    // Followed by count * (ChildEntry + variable-length name)
}

/// Wire format for Read request
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReadRequest {
    /// Node ID to read from
    pub node_id: u64,
    /// Byte offset within file
    pub offset: u64,
    /// Number of bytes to read
    pub length: u32,
}

/// Wire format for Read response header
/// Followed by the actual data bytes
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReadResponse {
    /// Number of bytes that follow this header
    pub data_len: u32,
}

/// Wire format for GetMetadata request
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GetMetadataRequest {
    /// Node ID to get metadata for
    pub node_id: u64,
}

/// Wire format for GetMetadata response
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GetMetadataResponse {
    /// Node kind
    pub kind: u8,
    /// Name length in bytes
    pub name_len: u8,
    /// Reserved for alignment
    pub _reserved: [u8; 6],
    /// File size (0 for directories)
    pub size: u64,
    /// Optional modification time (Unix timestamp seconds, 0 if unknown)
    pub mtime: u64,
    // Followed by name_len bytes of UTF-8 name data
}

/// Wire format for Error response
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ErrorResponse {
    /// Error code (see TreeProviderError)
    pub error_code: u8,
    /// Reserved for future use
    pub _reserved: [u8; 3],
}

/// Maximum size for a single name in bytes (for practical buffer allocation)
pub const MAX_NAME_LEN: usize = 255;

/// Maximum number of children returned in a single List response
/// (Clients should handle pagination if needed)
pub const MAX_CHILDREN_PER_LIST: usize = 64;

/// Maximum read size per request (to prevent DoS via large reads)
pub const MAX_READ_SIZE: usize = 65536; // 64 KB
