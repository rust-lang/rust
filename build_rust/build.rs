// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Collect build environment info and pass on to configure.rs
//!
//! The only two pieces of information we currently collect are
//! the triple spec (arch-vendor-os-abi) of the build machine
//! and the manifest directory of this Cargo package. The latter
//! is used to determine where the Rust source code is.
//!
//! Once finished collecting said variables this script will
//! write them into the file build_env.rs which configure.rs
//! will then include it (at compile time).

struct BuildEnv {
    build_triple : String,
    manifest_dir : String
}

/// On error, we will convert all errors into an error string,
/// display the error string and then panic.
struct ErrMsg {
    msg : String
}

impl<T : std::error::Error> std::convert::From<T> for ErrMsg {
    fn from(err : T) -> ErrMsg {
        ErrMsg { msg : err.description().to_string() }
    }
}

type Result<T> = std::result::Result<T, ErrMsg>;

/// We use the host triple of the compiler used to compile
/// this build script as the triple spec of the build machine.
/// This script must be compiled as a native executable for it
/// runs on the same machine as it is compiled.
fn get_build_env_info() -> Result<BuildEnv> {
    let host_triple = try!(std::env::var("HOST"));
    let target_triple = try!(std::env::var("TARGET"));
    if host_triple != target_triple {
        return Err(ErrMsg { msg : "The Rust Build System must be built as a native executable".to_string() });
    }
    let manifest_dir = try!(std::env::var("CARGO_MANIFEST_DIR"));
    Ok(BuildEnv {
        build_triple : host_triple.to_string(),
        manifest_dir : manifest_dir
    })
}

fn write_to_build_env_rs(info : &BuildEnv) -> Result<()> {
    use std::io::Write;
    let out_dir = env!("OUT_DIR");
    let dest_path = std::path::Path::new(&out_dir).join("build_env.rs");
    let mut f = try!(std::fs::File::create(&dest_path));
    try!(write!(&mut f,
                "const BUILD_TRIPLE : &'static str = \"{}\";
const MANIFEST_DIR : &'static str = r\"{}\";",
                info.build_triple,
                info.manifest_dir));
    Ok(())
}

fn run() -> Result<()> {
    let info = try!(get_build_env_info());
    write_to_build_env_rs(&info)
}

fn main() {
    match run() {
        Err(e) => {
            println!("Failed to collect build environment information: {}", e.msg);
            std::process::exit(1);
        },
        _ => {}
    }
}
