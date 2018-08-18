// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains utilities for outputting metadata for diagnostic errors.
//!
//! Each set of errors is mapped to a metadata file by a name, which is
//! currently always a crate name.

use std::collections::BTreeMap;
use std::env;
use std::fs::{remove_file, create_dir_all, File};
use std::io::Write;
use std::path::PathBuf;
use std::error::Error;
use rustc_serialize::json::as_json;

use syntax_pos::{Span, FileName};
use ext::base::ExtCtxt;
use diagnostics::plugin::{ErrorMap, ErrorInfo};

/// JSON encodable/decodable version of `ErrorInfo`.
#[derive(PartialEq, RustcDecodable, RustcEncodable)]
pub struct ErrorMetadata {
    pub description: Option<String>,
    pub use_site: Option<ErrorLocation>
}

/// Mapping from error codes to metadata that can be (de)serialized.
pub type ErrorMetadataMap = BTreeMap<String, ErrorMetadata>;

/// JSON encodable error location type with filename and line number.
#[derive(PartialEq, RustcDecodable, RustcEncodable)]
pub struct ErrorLocation {
    pub filename: FileName,
    pub line: usize
}

impl ErrorLocation {
    /// Create an error location from a span.
    pub fn from_span(ecx: &ExtCtxt, sp: Span) -> ErrorLocation {
        let loc = ecx.source_map().lookup_char_pos_adj(sp.lo());
        ErrorLocation {
            filename: loc.filename,
            line: loc.line
        }
    }
}

/// Get the directory where metadata for a given `prefix` should be stored.
///
/// See `output_metadata`.
pub fn get_metadata_dir(prefix: &str) -> PathBuf {
    env::var_os("RUSTC_ERROR_METADATA_DST")
        .map(PathBuf::from)
        .expect("env var `RUSTC_ERROR_METADATA_DST` isn't set")
        .join(prefix)
}

/// Map `name` to a path in the given directory: <directory>/<name>.json
fn get_metadata_path(directory: PathBuf, name: &str) -> PathBuf {
    directory.join(format!("{}.json", name))
}

/// Write metadata for the errors in `err_map` to disk, to a file corresponding to `prefix/name`.
///
/// For our current purposes the prefix is the target architecture and the name is a crate name.
/// If an error occurs steps will be taken to ensure that no file is created.
pub fn output_metadata(ecx: &ExtCtxt, prefix: &str, name: &str, err_map: &ErrorMap)
    -> Result<(), Box<dyn Error>>
{
    // Create the directory to place the file in.
    let metadata_dir = get_metadata_dir(prefix);
    create_dir_all(&metadata_dir)?;

    // Open the metadata file.
    let metadata_path = get_metadata_path(metadata_dir, name);
    let mut metadata_file = File::create(&metadata_path)?;

    // Construct a serializable map.
    let json_map = err_map.iter().map(|(k, &ErrorInfo { description, use_site })| {
        let key = k.as_str().to_string();
        let value = ErrorMetadata {
            description: description.map(|n| n.as_str().to_string()),
            use_site: use_site.map(|sp| ErrorLocation::from_span(ecx, sp))
        };
        (key, value)
    }).collect::<ErrorMetadataMap>();

    // Write the data to the file, deleting it if the write fails.
    let result = write!(&mut metadata_file, "{}", as_json(&json_map));
    if result.is_err() {
        remove_file(&metadata_path)?;
    }
    Ok(result?)
}
