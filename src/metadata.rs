//! Writing of the rustc metadata for dylibs

use object::write::{Object, StandardSegment, Symbol, SymbolSection};
use object::{SectionKind, SymbolFlags, SymbolKind, SymbolScope};

use rustc_metadata::EncodedMetadata;
use rustc_middle::ty::TyCtxt;

// Adapted from https://github.com/rust-lang/rust/blob/da573206f87b5510de4b0ee1a9c044127e409bd3/src/librustc_codegen_llvm/base.rs#L47-L112
pub(crate) fn new_metadata_object(
    tcx: TyCtxt<'_>,
    cgu_name: &str,
    metadata: &EncodedMetadata,
) -> Vec<u8> {
    use snap::write::FrameEncoder;
    use std::io::Write;

    let mut compressed = rustc_metadata::METADATA_HEADER.to_vec();
    FrameEncoder::new(&mut compressed).write_all(metadata.raw_data()).unwrap();

    let triple = crate::target_triple(tcx.sess);

    let binary_format = match triple.binary_format {
        target_lexicon::BinaryFormat::Elf => object::BinaryFormat::Elf,
        target_lexicon::BinaryFormat::Coff => object::BinaryFormat::Coff,
        target_lexicon::BinaryFormat::Macho => object::BinaryFormat::MachO,
        binary_format => tcx.sess.fatal(&format!("binary format {} is unsupported", binary_format)),
    };
    let architecture = match triple.architecture {
        target_lexicon::Architecture::Aarch64(_) => object::Architecture::Aarch64,
        target_lexicon::Architecture::Arm(_) => object::Architecture::Arm,
        target_lexicon::Architecture::Avr => object::Architecture::Avr,
        target_lexicon::Architecture::Hexagon => object::Architecture::Hexagon,
        target_lexicon::Architecture::Mips32(_) => object::Architecture::Mips,
        target_lexicon::Architecture::Mips64(_) => object::Architecture::Mips64,
        target_lexicon::Architecture::Msp430 => object::Architecture::Msp430,
        target_lexicon::Architecture::Powerpc => object::Architecture::PowerPc,
        target_lexicon::Architecture::Powerpc64 => object::Architecture::PowerPc64,
        target_lexicon::Architecture::Powerpc64le => todo!(),
        target_lexicon::Architecture::Riscv32(_) => object::Architecture::Riscv32,
        target_lexicon::Architecture::Riscv64(_) => object::Architecture::Riscv64,
        target_lexicon::Architecture::S390x => object::Architecture::S390x,
        target_lexicon::Architecture::Sparc64 => object::Architecture::Sparc64,
        target_lexicon::Architecture::Sparcv9 => object::Architecture::Sparc64,
        target_lexicon::Architecture::X86_32(_) => object::Architecture::I386,
        target_lexicon::Architecture::X86_64 => object::Architecture::X86_64,
        architecture => {
            tcx.sess.fatal(&format!("target architecture {:?} is unsupported", architecture,))
        }
    };
    let endian = match triple.endianness().unwrap() {
        target_lexicon::Endianness::Little => object::Endianness::Little,
        target_lexicon::Endianness::Big => object::Endianness::Big,
    };

    let mut object = Object::new(binary_format, architecture, endian);
    object.add_file_symbol(cgu_name.as_bytes().to_vec());

    let segment = object.segment_name(StandardSegment::Data).to_vec();
    let section_id = object.add_section(segment, b".rustc".to_vec(), SectionKind::Data);
    let offset = object.append_section_data(section_id, &compressed, 1);
    // For MachO and probably PE this is necessary to prevent the linker from throwing away the
    // .rustc section. For ELF this isn't necessary, but it also doesn't harm.
    object.add_symbol(Symbol {
        name: rustc_middle::middle::exported_symbols::metadata_symbol_name(tcx).into_bytes(),
        value: offset,
        size: compressed.len() as u64,
        kind: SymbolKind::Data,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Section(section_id),
        flags: SymbolFlags::None,
    });

    object.write().unwrap()
}
