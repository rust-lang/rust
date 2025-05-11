//! Provides the `link!` macro used by the generated windows bindings.
//!
//! This is a simple wrapper around an `extern` block with a `#[link]` attribute.
//! It's very roughly equivalent to the windows-targets crate.
#![no_std]
#![no_core]
#![feature(decl_macro)]
#![feature(no_core)]

#[cfg(feature = "windows_raw_dylib")]
pub macro link {
    ($library:literal $abi:literal $($link_name:literal)? $(#[$doc:meta])? fn $($function:tt)*) => (
        #[cfg_attr(not(target_arch = "x86"), link(name = $library, kind = "raw-dylib", modifiers = "+verbatim"))]
        #[cfg_attr(target_arch = "x86", link(name = $library, kind = "raw-dylib", modifiers = "+verbatim", import_name_type = "undecorated"))]
        unsafe extern $abi {
            $(#[link_name=$link_name])?
            pub fn $($function)*;
        }
    )
}
#[cfg(not(feature = "windows_raw_dylib"))]
pub macro link {
    ($library:literal $abi:literal $($link_name:literal)? $(#[$doc:meta])? fn $($function:tt)*) => (
        // Note: the windows-targets crate uses a pre-built Windows.lib import library which we don't
        // have in this repo. So instead we always link kernel32.lib and add the rest of the import
        // libraries below by using an empty extern block. This works because extern blocks are not
        // connected to the library given in the #[link] attribute.
        #[link(name = "kernel32")]
        unsafe extern $abi {
            $(#[link_name=$link_name])?
            pub fn $($function)*;
        }
    )
}

#[cfg(not(feature = "windows_raw_dylib"))]
#[cfg(not(target_os = "cygwin"))] // Cygwin doesn't need these libs
#[cfg_attr(target_vendor = "win7", link(name = "advapi32"))]
#[link(name = "ntdll")]
#[link(name = "userenv")]
#[link(name = "ws2_32")]
#[link(name = "dbghelp")] // required for backtrace-rs symbolization
unsafe extern "C" {}
