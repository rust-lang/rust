pub macro link {
    ($library:literal $abi:literal $($link_name:literal)? $(#[$doc:meta])? fn $($function:tt)*) => (
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
