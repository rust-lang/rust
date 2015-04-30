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
use std::path::PathBuf;
use std::fs::{read_dir, create_dir_all, OpenOptions, File};
use std::io::{Read, Write};
use std::error::Error;
use rustc_serialize::json::{self, as_json};

use codemap::Span;
use ext::base::ExtCtxt;
use diagnostics::plugin::{ErrorMap, ErrorInfo};

pub use self::Uniqueness::*;

// Default metadata directory to use for extended error JSON.
const ERROR_METADATA_DIR_DEFAULT: &'static str = "tmp/extended-errors";

// The name of the environment variable that sets the metadata dir.
const ERROR_METADATA_VAR: &'static str = "ERROR_METADATA_DIR";

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
    pub filename: String,
    pub line: usize
}

impl ErrorLocation {
    /// Create an error location from a span.
    pub fn from_span(ecx: &ExtCtxt, sp: Span) -> ErrorLocation {
        let loc = ecx.codemap().lookup_char_pos_adj(sp.lo);
        ErrorLocation {
            filename: loc.filename,
            line: loc.line
        }
    }
}

/// Type for describing the uniqueness of a set of error codes, as returned by `check_uniqueness`.
pub enum Uniqueness {
    /// All errors in the set checked are unique according to the metadata files checked.
    Unique,
    /// One or more errors in the set occur in another metadata file.
    /// This variant contains the first duplicate error code followed by the name
    /// of the metadata file where the duplicate appears.
    Duplicate(String, String)
}

/// Get the directory where metadata files should be stored.
pub fn get_metadata_dir() -> PathBuf {
    match env::var(ERROR_METADATA_VAR) {
        Ok(v) => From::from(v),
        Err(_) => From::from(ERROR_METADATA_DIR_DEFAULT)
    }
}

/// Get the path where error metadata for the set named by `name` should be stored.
fn get_metadata_path(name: &str) -> PathBuf {
    get_metadata_dir().join(format!("{}.json", name))
}

/// Check that the errors in `err_map` aren't present in any metadata files in the
/// metadata directory except the metadata file corresponding to `name`.
pub fn check_uniqueness(name: &str, err_map: &ErrorMap) -> Result<Uniqueness, Box<Error>> {
    let metadata_dir = get_metadata_dir();
    let metadata_path = get_metadata_path(name);

    // Create the error directory if it does not exist.
    try!(create_dir_all(&metadata_dir));

    // Check each file in the metadata directory.
    for entry in try!(read_dir(&metadata_dir)) {
        let path = try!(entry).path();

        // Skip any existing file for this set.
        if path == metadata_path {
            continue;
        }

        // Read the metadata file into a string.
        let mut metadata_str = String::new();
        try!(
            File::open(&path).and_then(|mut f|
            f.read_to_string(&mut metadata_str))
        );

        // Parse the JSON contents.
        let metadata: ErrorMetadataMap = try!(json::decode(&metadata_str));

        // Check for duplicates.
        for err in err_map.keys() {
            let err_code = err.as_str();
            if metadata.contains_key(err_code) {
                return Ok(Duplicate(
                    err_code.to_string(),
                    path.to_string_lossy().into_owned()
                ));
            }
        }
    }

    Ok(Unique)
}

/// Write metadata for the errors in `err_map` to disk, to a file corresponding to `name`.
pub fn output_metadata(ecx: &ExtCtxt, name: &str, err_map: &ErrorMap)
    -> Result<(), Box<Error>>
{
    let metadata_path = get_metadata_path(name);

    // Open the dump file.
    let mut dump_file = try!(OpenOptions::new()
        .write(true)
        .create(true)
        .open(&metadata_path)
    );

    // Construct a serializable map.
    let json_map = err_map.iter().map(|(k, &ErrorInfo { description, use_site })| {
        let key = k.as_str().to_string();
        let value = ErrorMetadata {
            description: description.map(|n| n.as_str().to_string()),
            use_site: use_site.map(|sp| ErrorLocation::from_span(ecx, sp))
        };
        (key, value)
    }).collect::<ErrorMetadataMap>();

    try!(write!(&mut dump_file, "{}", as_json(&json_map)));
    Ok(())
}
