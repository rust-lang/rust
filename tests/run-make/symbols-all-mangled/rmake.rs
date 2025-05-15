// Check that all symbols in cdylibs, staticlibs and bins are mangled
//@ only-elf some object file formats create multiple symbols for each function with different names
//@ ignore-nvptx64 (needs target std)
//@ ignore-cross-compile (host-only)

use run_make_support::object::read::{Object, ObjectSymbol};
use run_make_support::{bin_name, dynamic_lib_name, object, rfs, rustc, static_lib_name};

fn main() {
    let staticlib_name = static_lib_name("a_lib");
    let cdylib_name = dynamic_lib_name("a_lib");
    let exe_name = bin_name("an_executable");
    rustc().crate_type("cdylib").input("a_lib.rs").run();
    rustc().crate_type("staticlib").input("a_lib.rs").run();
    rustc().crate_type("bin").input("an_executable.rs").run();

    symbols_check_archive(&staticlib_name);
    symbols_check(&cdylib_name);
    symbols_check(&exe_name);
}

fn symbols_check_archive(path: &str) {
    let binary_data = rfs::read(path);
    let file = object::read::archive::ArchiveFile::parse(&*binary_data).unwrap();
    for symbol in file.symbols().unwrap().unwrap() {
        let symbol = symbol.unwrap();
        let name = strip_underscore_if_apple(std::str::from_utf8(symbol.name()).unwrap());
        if name.starts_with("_ZN") || name.starts_with("_R") {
            continue; // Correctly mangled
        }

        let member_name =
            std::str::from_utf8(file.member(symbol.offset()).unwrap().name()).unwrap();
        if !member_name.ends_with(".rcgu.o") || member_name.contains("compiler_builtins") {
            continue; // All compiler-builtins symbols must remain unmangled
        }

        if name.contains("rust_eh_personality") {
            continue; // Unfortunately LLVM doesn't allow us to mangle this symbol
        }

        if name.contains(".llvm.") {
            // Starting in LLVM 21 we get various implementation-detail functions which
            // contain .llvm. that are not a problem.
            continue;
        }

        panic!("Unmangled symbol found in {path}: {name}");
    }
}

fn symbols_check(path: &str) {
    let binary_data = rfs::read(path);
    let file = object::File::parse(&*binary_data).unwrap();
    for symbol in file.symbols() {
        if !symbol.is_definition() || !symbol.is_global() {
            continue;
        }
        if symbol.is_weak() {
            continue; // Likely an intrinsic from compiler-builtins
        }
        let name = strip_underscore_if_apple(symbol.name().unwrap());
        if name.starts_with("_ZN") || name.starts_with("_R") {
            continue; // Correctly mangled
        }

        if !name.contains("rust") {
            // Assume that this symbol doesn't originate from rustc. This may
            // be wrong, but even if so symbol_check_archive will likely
            // catch it.
            continue;
        }

        if name.contains("rust_eh_personality") {
            continue; // Unfortunately LLVM doesn't allow us to mangle this symbol
        }

        if name.contains(".llvm.") {
            // Starting in LLVM 21 we get various implementation-detail functions which
            // contain .llvm. that are not a problem.
            continue;
        }

        panic!("Unmangled symbol found in {path}: {name}");
    }
}

fn strip_underscore_if_apple(symbol: &str) -> &str {
    if cfg!(target_vendor = "apple") { symbol.strip_prefix("_").unwrap() } else { symbol }
}
