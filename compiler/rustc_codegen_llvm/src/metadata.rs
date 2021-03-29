use rustc_target::spec::Target;

use rustc_fs_util::path_to_c_string;
use std::path::Path;
use std::slice;

pub fn metadata_section_name(target: &Target) -> &'static str {
    // Historical note:
    //
    // When using link.exe it was seen that the section name `.note.rustc`
    // was getting shortened to `.note.ru`, and according to the PE and COFF
    // specification:
    //
    // > Executable images do not use a string table and do not support
    // > section names longer than 8 characters
    //
    // https://docs.microsoft.com/en-us/windows/win32/debug/pe-format
    //
    // As a result, we choose a slightly shorter name! As to why
    // `.note.rustc` works on MinGW, that's another good question...

    if target.is_like_osx { "__DATA,.rustc" } else { ".rustc" }
}
