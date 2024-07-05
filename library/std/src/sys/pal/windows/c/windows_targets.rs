//! Provides the `link!` macro used by the generated windows bindings.
//!
//! This is a simple wrapper around an `extern` block with a `#[link]` attribute.
//! It's very roughly equivalent to the windows-targets crate.

pub macro link {
    ($library:literal $abi:literal $($link_name:literal)? $(#[$doc:meta])? fn $($function:tt)*) => (
        // Note: the windows-targets crate uses a pre-built Windows.lib import library which we don't
        // have in this repo. So instead we always link kernel32.lib and add the rest of the import
        // libraries below by using an empty extern block. This works because extern blocks are not
        // connected to the library given in the #[link] attribute.
        #[link(name = "kernel32")]
        extern $abi {
            $(#[link_name=$link_name])?
            pub fn $($function)*;
        }
    )
}

#[link(name = "advapi32")]
#[link(name = "ntdll")]
#[link(name = "userenv")]
#[link(name = "ws2_32")]
extern "C" {}
