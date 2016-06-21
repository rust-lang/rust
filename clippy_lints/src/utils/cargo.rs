use std::collections::HashMap;
use std::process::Command;
use std::str::{from_utf8, Utf8Error};
use std::io;
use rustc_serialize::json;

#[derive(RustcDecodable, Debug)]
pub struct Metadata {
    pub packages: Vec<Package>,
    resolve: Option<()>,
    pub version: usize,
}

#[derive(RustcDecodable, Debug)]
pub struct Package {
    pub name: String,
    pub version: String,
    id: String,
    source: Option<String>,
    pub dependencies: Vec<Dependency>,
    pub targets: Vec<Target>,
    features: HashMap<String, Vec<String>>,
    manifest_path: String,
}

#[derive(RustcDecodable, Debug)]
pub struct Dependency {
    pub name: String,
    source: Option<String>,
    pub req: String,
    kind: Option<String>,
    optional: bool,
    uses_default_features: bool,
    features: Vec<String>,
    target: Option<String>,
}

#[derive(RustcDecodable, Debug)]
pub struct Target {
    pub name: String,
    pub kind: Vec<String>,
    src_path: String,
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Utf8(Utf8Error),
    Json(json::DecoderError),
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}
impl From<Utf8Error> for Error {
    fn from(err: Utf8Error) -> Self {
        Error::Utf8(err)
    }
}
impl From<json::DecoderError> for Error {
    fn from(err: json::DecoderError) -> Self {
        Error::Json(err)
    }
}

pub fn metadata() -> Result<Metadata, Error> {
    let output = Command::new("cargo").args(&["metadata", "--no-deps"]).output()?;
    let stdout = from_utf8(&output.stdout)?;
    Ok(json::decode(stdout)?)
}
