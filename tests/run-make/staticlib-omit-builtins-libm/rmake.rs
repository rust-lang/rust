// compiler-builtins weakly defines the f32/f64 math symbols (its `full_availability` list) so
// `no_std` targets without a system libm still work. A staticlib that records `-lm` must omit
// them so libm's strong definitions win at the final link (#142119); the no-`-lm` control keeps
// its weak fallbacks. `only-gnu` because only gnu libm (glibc) provides these symbols.

//@ only-gnu
//@ ignore-cross-compile

use std::collections::HashSet;

use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::read::elf::{FileHeader as _, SectionHeader as _, Sym as _};
use run_make_support::object::{Endianness, elf};
use run_make_support::{rfs, rustc, static_lib_name};

// Keep in sync with `COMPILER_BUILTINS_LIBM_SYMBOLS` in rustc_codegen_ssa/src/back/link.rs.
const LIBM_SYMBOLS: &[&str] = &[
    "cbrtf",
    "ceilf",
    "copysignf",
    "fabsf",
    "fdimf",
    "floorf",
    "fmaf",
    "fmaxf",
    "fminf",
    "fmodf",
    "rintf",
    "roundf",
    "sqrtf",
    "truncf",
    "cbrt",
    "ceil",
    "copysign",
    "fabs",
    "fdim",
    "floor",
    "fma",
    "fmax",
    "fmin",
    "fmod",
    "rint",
    "round",
    "sqrt",
    "trunc",
];

fn main() {
    rustc().input("no_libm.rs").crate_type("staticlib").panic("abort").run();
    let no_libm = static_lib_name("no_libm");
    let no_libm_defined = defined_global_symbols(&no_libm);
    for name in LIBM_SYMBOLS {
        assert!(
            no_libm_defined.contains(*name),
            "expected weak fallback `{name}` to be kept without `-lm` in `{}`",
            no_libm
        );
    }

    rustc().input("lib.rs").crate_type("staticlib").panic("abort").run();
    let with_libm = static_lib_name("lib");
    let with_libm_defined = defined_global_symbols(&with_libm);
    for name in LIBM_SYMBOLS {
        assert!(
            !with_libm_defined.contains(*name),
            "weak definition `{name}` should be omitted when `-lm` is recorded in `{with_libm}`"
        );
    }

    // Not a blanket strip: the f16/f128 and integer fallbacks survive. `ceilf16` also guards
    // against matching `ceilf` as a prefix of `ceilf16`.
    assert!(
        with_libm_defined.contains("ceilf16"),
        "compiler-builtins' f16 fallback must survive the omit in `{with_libm}`"
    );
    assert!(
        with_libm_defined.contains("__floatdidf"),
        "compiler-builtins' integer fallback must survive the omit in `{with_libm}`"
    );
}

/// Every defined global/weak symbol in the archive. The omitted symbols can still appear as
/// undefined references (e.g. `use_mathf` calling `ceilf`), so filter `SHN_UNDEF` rather than
/// checking mere presence.
fn defined_global_symbols(archive_path: &str) -> HashSet<String> {
    let archive_data = rfs::read(archive_path);
    let archive = ArchiveFile::parse(archive_data.as_slice()).unwrap();
    let mut defined = HashSet::new();

    for member in archive.members() {
        let member = member.unwrap();
        let data = member.data(archive_data.as_slice()).unwrap();

        if let Ok(header) = elf::FileHeader64::<Endianness>::parse(data) {
            collect_elf_defined(header, data, &mut defined);
        } else if let Ok(header) = elf::FileHeader32::<Endianness>::parse(data) {
            collect_elf_defined(header, data, &mut defined);
        }
    }

    defined
}

fn collect_elf_defined<
    Elf: run_make_support::object::read::elf::FileHeader<Endian = Endianness>,
>(
    header: &Elf,
    data: &[u8],
    defined: &mut HashSet<String>,
) {
    let Ok(endian) = header.endian() else { return };
    let Ok(sections) = header.sections(endian, data) else { return };

    for (si, section) in sections.enumerate() {
        if section.sh_type(endian) != elf::SHT_SYMTAB {
            continue;
        }
        let Ok(symbols) = run_make_support::object::read::elf::SymbolTable::parse(
            endian, data, &sections, si, section,
        ) else {
            continue;
        };
        let strtab = symbols.strings();

        for symbol in symbols.symbols() {
            let bind = symbol.st_bind();
            if bind != elf::STB_GLOBAL && bind != elf::STB_WEAK {
                continue;
            }
            if symbol.st_shndx(endian) == elf::SHN_UNDEF {
                continue;
            }
            let Ok(name_bytes) = symbol.name(endian, strtab) else { continue };
            if let Ok(name) = str::from_utf8(name_bytes) {
                defined.insert(name.to_string());
            }
        }
    }
}
