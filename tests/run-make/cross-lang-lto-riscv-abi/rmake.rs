//! Make sure that cross-language LTO works on riscv targets, which requires extra `target-abi`
//! metadata to be emitted.
//@ needs-force-clang-based-tests
//@ needs-llvm-components: riscv

// ignore-tidy-linelength

use object::elf;
use object::read::elf as readelf;
use run_make_support::{bin_name, clang, object, rfs, rustc};

fn check_target<H: readelf::FileHeader<Endian = object::Endianness>>(
    target: &str,
    clang_target: &str,
    carch: &str,
    is_double_float: bool,
) {
    eprintln!("Checking target {target}");
    // Rust part
    rustc()
        .input("riscv-xlto.rs")
        .crate_type("rlib")
        .target(target)
        .panic("abort")
        .linker_plugin_lto("on")
        .run();
    // C part
    clang()
        .target(clang_target)
        .arch(carch)
        .lto("thin")
        .use_ld("lld")
        .no_stdlib()
        .out_exe("riscv-xlto")
        .input("cstart.c")
        .input("libriscv_xlto.rlib")
        .run();

    // Check that the built binary has correct float abi
    let executable = bin_name("riscv-xlto");
    let data = rfs::read(&executable);
    let header = <H>::parse(&*data).unwrap();
    let endian = match header.e_ident().data {
        elf::ELFDATA2LSB => object::Endianness::Little,
        elf::ELFDATA2MSB => object::Endianness::Big,
        x => unreachable!("invalid e_ident data: {:#010b}", x),
    };
    // Check `(e_flags & EF_RISCV_FLOAT_ABI) == EF_RISCV_FLOAT_ABI_DOUBLE`.
    //
    // See
    // <https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc#elf-object-files>.
    if is_double_float {
        assert_eq!(
            header.e_flags(endian) & elf::EF_RISCV_FLOAT_ABI,
            elf::EF_RISCV_FLOAT_ABI_DOUBLE,
            "expected {target} to use double ABI, but it did not"
        );
    } else {
        assert_ne!(
            header.e_flags(endian) & elf::EF_RISCV_FLOAT_ABI,
            elf::EF_RISCV_FLOAT_ABI_DOUBLE,
            "did not expected {target} to use double ABI"
        );
    }
}

fn main() {
    check_target::<elf::FileHeader64<object::Endianness>>(
        "riscv64gc-unknown-linux-gnu",
        "riscv64-linux-gnu",
        "rv64gc",
        true,
    );
    check_target::<elf::FileHeader64<object::Endianness>>(
        "riscv64imac-unknown-none-elf",
        "riscv64-unknown-elf",
        "rv64imac",
        false,
    );
    check_target::<elf::FileHeader32<object::Endianness>>(
        "riscv32imac-unknown-none-elf",
        "riscv32-unknown-elf",
        "rv32imac",
        false,
    );
    check_target::<elf::FileHeader32<object::Endianness>>(
        "riscv32gc-unknown-linux-gnu",
        "riscv32-linux-gnu",
        "rv32gc",
        true,
    );
}
