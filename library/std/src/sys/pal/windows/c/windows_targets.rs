#[cfg(feature = "windows_raw_dylib")]
pub macro link {
    ($library:literal $abi:literal $($link_name:literal)? $(#[$doc:meta])? fn $($function:tt)*) => (
        #[cfg_attr(not(target_arch = "x86"), link(name = $library, kind = "raw-dylib", modifiers = "+verbatim"))]
        #[cfg_attr(target_arch = "x86", link(name = $library, kind = "raw-dylib", modifiers = "+verbatim", import_name_type = "undecorated"))]
        extern $abi {
            $(#[link_name=$link_name])?
            pub fn $($function)*;
        }
    )
}
#[cfg(not(feature = "windows_raw_dylib"))]
pub macro link {
    ($library:literal $abi:literal $($link_name:literal)? $(#[$doc:meta])? fn $($function:tt)*) => (
        #[link(name = "kernel32")]
        extern $abi {
            $(#[link_name=$link_name])?
            pub fn $($function)*;
        }
    )
}

#[cfg(not(feature = "windows_raw_dylib"))]
#[link(name = "advapi32")]
#[link(name = "ntdll")]
#[link(name = "userenv")]
#[link(name = "ws2_32")]
extern "C" {}
