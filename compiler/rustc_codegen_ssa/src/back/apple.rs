use rustc_middle::middle::exported_symbols::SymbolExportKind;
use rustc_target::spec::Target;
pub(super) use rustc_target::spec::apple::OSVersion;

#[cfg(test)]
mod tests;

pub(super) fn macho_platform(target: &Target) -> u32 {
    match (&*target.os, &*target.abi) {
        ("macos", _) => object::macho::PLATFORM_MACOS,
        ("ios", "macabi") => object::macho::PLATFORM_MACCATALYST,
        ("ios", "sim") => object::macho::PLATFORM_IOSSIMULATOR,
        ("ios", _) => object::macho::PLATFORM_IOS,
        ("watchos", "sim") => object::macho::PLATFORM_WATCHOSSIMULATOR,
        ("watchos", _) => object::macho::PLATFORM_WATCHOS,
        ("tvos", "sim") => object::macho::PLATFORM_TVOSSIMULATOR,
        ("tvos", _) => object::macho::PLATFORM_TVOS,
        ("visionos", "sim") => object::macho::PLATFORM_XROSSIMULATOR,
        ("visionos", _) => object::macho::PLATFORM_XROS,
        _ => unreachable!("tried to get Mach-O platform for non-Apple target"),
    }
}

/// Add relocation and section data needed for a symbol to be considered
/// undefined by ld64.
///
/// The relocation must be valid, and hence must point to a valid piece of
/// machine code, and hence this is unfortunately very architecture-specific.
///
///
/// # New architectures
///
/// The values here are basically the same as emitted by the following program:
///
/// ```c
/// // clang -c foo.c -target $CLANG_TARGET
/// void foo(void);
///
/// extern int bar;
///
/// void* foobar[2] = {
///     (void*)foo,
///     (void*)&bar,
///     // ...
/// };
/// ```
///
/// Can be inspected with:
/// ```console
/// objdump --macho --reloc foo.o
/// objdump --macho --full-contents foo.o
/// ```
pub(super) fn add_data_and_relocation(
    file: &mut object::write::Object<'_>,
    section: object::write::SectionId,
    symbol: object::write::SymbolId,
    target: &Target,
    kind: SymbolExportKind,
) -> object::write::Result<()> {
    let authenticated_pointer =
        kind == SymbolExportKind::Text && target.llvm_target.starts_with("arm64e");

    let data: &[u8] = match target.pointer_width {
        _ if authenticated_pointer => &[0, 0, 0, 0, 0, 0, 0, 0x80],
        32 => &[0; 4],
        64 => &[0; 8],
        pointer_width => unimplemented!("unsupported Apple pointer width {pointer_width:?}"),
    };

    if target.arch == "x86_64" {
        // Force alignment for the entire section to be 16 on x86_64.
        file.section_mut(section).append_data(&[], 16);
    } else {
        // Elsewhere, the section alignment is the same as the pointer width.
        file.section_mut(section).append_data(&[], target.pointer_width as u64);
    }

    let offset = file.section_mut(section).append_data(data, data.len() as u64);

    let flags = if authenticated_pointer {
        object::write::RelocationFlags::MachO {
            r_type: object::macho::ARM64_RELOC_AUTHENTICATED_POINTER,
            r_pcrel: false,
            r_length: 3,
        }
    } else if target.arch == "arm" {
        // FIXME(madsmtm): Remove once `object` supports 32-bit ARM relocations:
        // https://github.com/gimli-rs/object/pull/757
        object::write::RelocationFlags::MachO {
            r_type: object::macho::ARM_RELOC_VANILLA,
            r_pcrel: false,
            r_length: 2,
        }
    } else {
        object::write::RelocationFlags::Generic {
            kind: object::RelocationKind::Absolute,
            encoding: object::RelocationEncoding::Generic,
            size: target.pointer_width as u8,
        }
    };

    file.add_relocation(section, object::write::Relocation { offset, addend: 0, symbol, flags })?;

    Ok(())
}

pub(super) fn add_version_to_llvm_target(
    llvm_target: &str,
    deployment_target: OSVersion,
) -> String {
    let mut components = llvm_target.split("-");
    let arch = components.next().expect("apple target should have arch");
    let vendor = components.next().expect("apple target should have vendor");
    let os = components.next().expect("apple target should have os");
    let environment = components.next();
    assert_eq!(components.next(), None, "too many LLVM triple components");

    assert!(
        !os.contains(|c: char| c.is_ascii_digit()),
        "LLVM target must not already be versioned"
    );

    let version = deployment_target.fmt_full();
    if let Some(env) = environment {
        // Insert version into OS, before environment
        format!("{arch}-{vendor}-{os}{version}-{env}")
    } else {
        format!("{arch}-{vendor}-{os}{version}")
    }
}
