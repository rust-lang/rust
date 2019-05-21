#[cfg(target_os = "linux")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "linux";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "macos")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "macos";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".dylib";
    pub const DLL_EXTENSION: &str = "dylib";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "ios")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "ios";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".dylib";
    pub const DLL_EXTENSION: &str = "dylib";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "freebsd")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "freebsd";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "dragonfly")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "dragonfly";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "netbsd")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "netbsd";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "openbsd")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "openbsd";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "android")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "android";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "solaris")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "solaris";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "haiku")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "haiku";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(all(target_os = "emscripten", target_arch = "asmjs"))]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "emscripten";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = ".js";
    pub const EXE_EXTENSION: &str = "js";
}

#[cfg(all(target_os = "emscripten", target_arch = "wasm32"))]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "emscripten";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = ".js";
    pub const EXE_EXTENSION: &str = "js";
}

#[cfg(target_os = "fuchsia")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "fuchsia";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "l4re")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "l4re";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "hermit")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "hermit";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}
