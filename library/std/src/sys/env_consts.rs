//! Constants associated with each target.

// Replaces the #[else] gate with #[cfg(not(any(…)))] of all the other gates.
// This ensures that they must be mutually exclusive and do not have precedence
// like cfg_if!.
macro cfg_unordered(
    $(#[cfg($cfg:meta)] $os:item)*
    #[else] $fallback:item
) {
    $(#[cfg($cfg)] $os)*
    #[cfg(not(any($($cfg),*)))] $fallback
}

// Keep entries sorted alphabetically and mutually exclusive.

cfg_unordered! {

#[cfg(target_os = "aix")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "aix";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".a";
    pub const DLL_EXTENSION: &str = "a";
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

#[cfg(target_os = "cygwin")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "cygwin";
    pub const DLL_PREFIX: &str = "";
    pub const DLL_SUFFIX: &str = ".dll";
    pub const DLL_EXTENSION: &str = "dll";
    pub const EXE_SUFFIX: &str = ".exe";
    pub const EXE_EXTENSION: &str = "exe";
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

#[cfg(target_os = "emscripten")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "emscripten";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = ".js";
    pub const EXE_EXTENSION: &str = "js";
}

#[cfg(target_os = "espidf")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "espidf";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
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

#[cfg(target_os = "hermit")]
pub mod os {
    pub const FAMILY: &str = "";
    pub const OS: &str = "hermit";
    pub const DLL_PREFIX: &str = "";
    pub const DLL_SUFFIX: &str = "";
    pub const DLL_EXTENSION: &str = "";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "horizon")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "horizon";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = ".elf";
    pub const EXE_EXTENSION: &str = "elf";
}

#[cfg(target_os = "hurd")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "hurd";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "illumos")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "illumos";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
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

#[cfg(target_os = "nto")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "nto";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "nuttx")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "nuttx";
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

#[cfg(target_os = "redox")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "redox";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "rtems")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "rtems";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
pub mod os {
    pub const FAMILY: &str = "";
    pub const OS: &str = "";
    pub const DLL_PREFIX: &str = "";
    pub const DLL_SUFFIX: &str = ".sgxs";
    pub const DLL_EXTENSION: &str = "sgxs";
    pub const EXE_SUFFIX: &str = ".sgxs";
    pub const EXE_EXTENSION: &str = "sgxs";
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

#[cfg(target_os = "solid_asp3")]
pub mod os {
    pub const FAMILY: &str = "itron";
    pub const OS: &str = "solid";
    pub const DLL_PREFIX: &str = "";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "tvos")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "tvos";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".dylib";
    pub const DLL_EXTENSION: &str = "dylib";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "uefi")]
pub mod os {
    pub const FAMILY: &str = "";
    pub const OS: &str = "uefi";
    pub const DLL_PREFIX: &str = "";
    pub const DLL_SUFFIX: &str = "";
    pub const DLL_EXTENSION: &str = "";
    pub const EXE_SUFFIX: &str = ".efi";
    pub const EXE_EXTENSION: &str = "efi";
}

#[cfg(target_os = "visionos")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "visionos";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".dylib";
    pub const DLL_EXTENSION: &str = "dylib";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "vita")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "vita";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = ".elf";
    pub const EXE_EXTENSION: &str = "elf";
}

#[cfg(target_os = "vxworks")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "vxworks";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".so";
    pub const DLL_EXTENSION: &str = "so";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(all(target_family = "wasm", not(any(target_os = "emscripten", target_os = "linux"))))]
pub mod os {
    pub const FAMILY: &str = "";
    pub const OS: &str = "";
    pub const DLL_PREFIX: &str = "";
    pub const DLL_SUFFIX: &str = ".wasm";
    pub const DLL_EXTENSION: &str = "wasm";
    pub const EXE_SUFFIX: &str = ".wasm";
    pub const EXE_EXTENSION: &str = "wasm";
}

#[cfg(target_os = "watchos")]
pub mod os {
    pub const FAMILY: &str = "unix";
    pub const OS: &str = "watchos";
    pub const DLL_PREFIX: &str = "lib";
    pub const DLL_SUFFIX: &str = ".dylib";
    pub const DLL_EXTENSION: &str = "dylib";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

#[cfg(target_os = "windows")]
pub mod os {
    pub const FAMILY: &str = "windows";
    pub const OS: &str = "windows";
    pub const DLL_PREFIX: &str = "";
    pub const DLL_SUFFIX: &str = ".dll";
    pub const DLL_EXTENSION: &str = "dll";
    pub const EXE_SUFFIX: &str = ".exe";
    pub const EXE_EXTENSION: &str = "exe";
}

#[cfg(target_os = "zkvm")]
pub mod os {
    pub const FAMILY: &str = "";
    pub const OS: &str = "";
    pub const DLL_PREFIX: &str = "";
    pub const DLL_SUFFIX: &str = ".elf";
    pub const DLL_EXTENSION: &str = "elf";
    pub const EXE_SUFFIX: &str = ".elf";
    pub const EXE_EXTENSION: &str = "elf";
}

// The fallback when none of the other gates match.
#[else]
pub mod os {
    pub const FAMILY: &str = "";
    pub const OS: &str = "";
    pub const DLL_PREFIX: &str = "";
    pub const DLL_SUFFIX: &str = "";
    pub const DLL_EXTENSION: &str = "";
    pub const EXE_SUFFIX: &str = "";
    pub const EXE_EXTENSION: &str = "";
}

}
