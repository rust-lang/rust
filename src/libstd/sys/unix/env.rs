// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(target_os = "linux")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "linux";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "macos")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "macos";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".dylib";
    pub const DLL_EXTENSION: &'static str = "dylib";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "ios")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "ios";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".dylib";
    pub const DLL_EXTENSION: &'static str = "dylib";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "freebsd")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "freebsd";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "dragonfly")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "dragonfly";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "bitrig")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "bitrig";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "netbsd")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "netbsd";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "openbsd")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "openbsd";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "android")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "android";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "solaris")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "solaris";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(all(target_os = "nacl", not(target_arch = "le32")))]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "nacl";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = ".nexe";
    pub const EXE_EXTENSION: &'static str = "nexe";
}
#[cfg(all(target_os = "nacl", target_arch = "le32"))]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "pnacl";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".pso";
    pub const DLL_EXTENSION: &'static str = "pso";
    pub const EXE_SUFFIX: &'static str = ".pexe";
    pub const EXE_EXTENSION: &'static str = "pexe";
}

#[cfg(target_os = "haiku")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "haiku";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(all(target_os = "emscripten", target_arch = "asmjs"))]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "emscripten";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = ".js";
    pub const EXE_EXTENSION: &'static str = "js";
}

#[cfg(all(target_os = "emscripten", target_arch = "wasm32"))]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "emscripten";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = ".js";
    pub const EXE_EXTENSION: &'static str = "js";
}

#[cfg(target_os = "fuchsia")]
pub mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "fuchsia";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}
