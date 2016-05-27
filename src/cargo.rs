use std::collections::HashMap;

#[derive(RustcDecodable, Debug)]
pub struct Metadata {
    pub packages: Vec<Package>,
    resolve: Option<()>,
    pub version: usize,
}

#[derive(RustcDecodable, Debug)]
pub struct Package {
    name: String,
    version: String,
    id: String,
    source: Option<()>,
    dependencies: Vec<Dependency>,
    pub targets: Vec<Target>,
    features: HashMap<String, Vec<String>>,
    manifest_path: String,
}

#[derive(RustcDecodable, Debug)]
pub struct Dependency {
    name: String,
    source: Option<String>,
    req: String,
    kind: Option<String>,
    optional: bool,
    uses_default_features: bool,
    features: Vec<HashMap<String, String>>,
    target: Option<()>,
}

#[allow(non_camel_case_types)]
#[derive(RustcDecodable, Debug)]
pub enum Kind {
    dylib,
    test,
    bin,
    lib,
}

#[derive(RustcDecodable, Debug)]
pub struct Target {
    pub name: String,
    pub kind: Vec<Kind>,
    src_path: String,
}
