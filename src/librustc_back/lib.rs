// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Some stuff used by rustc that doesn't have many dependencies
//!
//! Originally extracted from rustc::back, which was nominally the
//! compiler 'backend', though LLVM is rustc's backend, so rustc_back
//! is really just odds-and-ends relating to code gen and linking.
//! This crate mostly exists to make rustc smaller, so we might put
//! more 'stuff' here in the future.  It does not have a dependency on
//! rustc_llvm.
//!
//! FIXME: Split this into two crates: one that has deps on syntax, and
//! one that doesn't; the one that doesn't might get decent parallel
//! build speedups.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(box_syntax)]
#![feature(const_fn)]
#![feature(fs_read_write)]

extern crate syntax;
extern crate rand;
extern crate serialize;
#[macro_use] extern crate log;

extern crate serialize as rustc_serialize; // used by deriving

pub mod target;

use std::str::FromStr;

use serialize::json::{Json, ToJson};

macro_rules! linker_flavor {
    ($(($variant:ident, $string:expr),)+) => {
        #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash,
                 RustcEncodable, RustcDecodable)]
        pub enum LinkerFlavor {
            $($variant,)+
        }

        impl LinkerFlavor {
            pub const fn one_of() -> &'static str {
                concat!("one of: ", $($string, " ",)+)
            }

            pub fn from_str(s: &str) -> Option<Self> {
                Some(match s {
                    $($string => LinkerFlavor::$variant,)+
                    _ => return None,
                })
            }

            pub fn desc(&self) -> &str {
                match *self {
                    $(LinkerFlavor::$variant => $string,)+
                }
            }
        }

        impl ToJson for LinkerFlavor {
            fn to_json(&self) -> Json {
                self.desc().to_json()
            }
        }
    }
}

linker_flavor! {
    (Em, "em"),
    (Binaryen, "binaryen"),
    (Gcc, "gcc"),
    (Ld, "ld"),
    (Msvc, "msvc"),
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub enum PanicStrategy {
    Unwind,
    Abort,
}

impl PanicStrategy {
    pub fn desc(&self) -> &str {
        match *self {
            PanicStrategy::Unwind => "unwind",
            PanicStrategy::Abort => "abort",
        }
    }
}

impl ToJson for PanicStrategy {
    fn to_json(&self) -> Json {
        match *self {
            PanicStrategy::Abort => "abort".to_json(),
            PanicStrategy::Unwind => "unwind".to_json(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub enum RelroLevel {
    Full,
    Partial,
    Off,
}

impl RelroLevel {
    pub fn desc(&self) -> &str {
        match *self {
            RelroLevel::Full => "full",
            RelroLevel::Partial => "partial",
            RelroLevel::Off => "off",
        }
    }
}

impl FromStr for RelroLevel {
    type Err = ();

    fn from_str(s: &str) -> Result<RelroLevel, ()> {
        match s {
            "full" => Ok(RelroLevel::Full),
            "partial" => Ok(RelroLevel::Partial),
            "off" => Ok(RelroLevel::Off),
            _ => Err(()),
        }
    }
}

impl ToJson for RelroLevel {
    fn to_json(&self) -> Json {
        match *self {
            RelroLevel::Full => "full".to_json(),
            RelroLevel::Partial => "partial".to_json(),
            RelroLevel::Off => "off".to_json(),
        }
    }
}
