//! Abstraction around the object writing crate

use std::convert::{TryFrom, TryInto};

use rustc_data_structures::fx::FxHashMap;
use rustc_session::Session;

use cranelift_module::FuncId;

use object::write::*;
use object::{RelocationEncoding, RelocationKind, SectionKind, SymbolFlags};

use cranelift_object::{ObjectBuilder, ObjectModule, ObjectProduct};

use gimli::SectionId;

use crate::debuginfo::{DebugReloc, DebugRelocName};

pub(crate) trait WriteMetadata {
    fn add_rustc_section(&mut self, symbol_name: String, data: Vec<u8>, is_like_osx: bool);
}

impl WriteMetadata for object::write::Object {
    fn add_rustc_section(&mut self, symbol_name: String, data: Vec<u8>, _is_like_osx: bool) {
        let segment = self
            .segment_name(object::write::StandardSegment::Data)
            .to_vec();
        let section_id = self.add_section(segment, b".rustc".to_vec(), object::SectionKind::Data);
        let offset = self.append_section_data(section_id, &data, 1);
        // For MachO and probably PE this is necessary to prevent the linker from throwing away the
        // .rustc section. For ELF this isn't necessary, but it also doesn't harm.
        self.add_symbol(object::write::Symbol {
            name: symbol_name.into_bytes(),
            value: offset,
            size: data.len() as u64,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Section(section_id),
            flags: SymbolFlags::None,
        });
    }
}

pub(crate) trait WriteDebugInfo {
    type SectionId: Copy;

    fn add_debug_section(&mut self, name: SectionId, data: Vec<u8>) -> Self::SectionId;
    fn add_debug_reloc(
        &mut self,
        section_map: &FxHashMap<SectionId, Self::SectionId>,
        from: &Self::SectionId,
        reloc: &DebugReloc,
    );
}

impl WriteDebugInfo for ObjectProduct {
    type SectionId = (object::write::SectionId, object::write::SymbolId);

    fn add_debug_section(
        &mut self,
        id: SectionId,
        data: Vec<u8>,
    ) -> (object::write::SectionId, object::write::SymbolId) {
        let name = if self.object.format() == object::BinaryFormat::MachO {
            id.name().replace('.', "__") // machO expects __debug_info instead of .debug_info
        } else {
            id.name().to_string()
        }
        .into_bytes();

        let segment = self.object.segment_name(StandardSegment::Debug).to_vec();
        // FIXME use SHT_X86_64_UNWIND for .eh_frame
        let section_id = self.object.add_section(
            segment,
            name,
            if id == SectionId::EhFrame {
                SectionKind::ReadOnlyData
            } else {
                SectionKind::Debug
            },
        );
        self.object
            .section_mut(section_id)
            .set_data(data, if id == SectionId::EhFrame { 8 } else { 1 });
        let symbol_id = self.object.section_symbol(section_id);
        (section_id, symbol_id)
    }

    fn add_debug_reloc(
        &mut self,
        section_map: &FxHashMap<SectionId, Self::SectionId>,
        from: &Self::SectionId,
        reloc: &DebugReloc,
    ) {
        let (symbol, symbol_offset) = match reloc.name {
            DebugRelocName::Section(id) => (section_map.get(&id).unwrap().1, 0),
            DebugRelocName::Symbol(id) => {
                let symbol_id = self.function_symbol(FuncId::from_u32(id.try_into().unwrap()));
                self.object
                    .symbol_section_and_offset(symbol_id)
                    .expect("Debug reloc for undef sym???")
            }
        };
        self.object
            .add_relocation(
                from.0,
                Relocation {
                    offset: u64::from(reloc.offset),
                    symbol,
                    kind: reloc.kind,
                    encoding: RelocationEncoding::Generic,
                    size: reloc.size * 8,
                    addend: i64::try_from(symbol_offset).unwrap() + reloc.addend,
                },
            )
            .unwrap();
    }
}

// FIXME remove once atomic instructions are implemented in Cranelift.
pub(crate) trait AddConstructor {
    fn add_constructor(&mut self, func_id: FuncId);
}

impl AddConstructor for ObjectProduct {
    fn add_constructor(&mut self, func_id: FuncId) {
        let symbol = self.function_symbol(func_id);
        let segment = self
            .object
            .segment_name(object::write::StandardSegment::Data);
        let init_array_section =
            self.object
                .add_section(segment.to_vec(), b".init_array".to_vec(), SectionKind::Data);
        let address_size = self
            .object
            .architecture()
            .address_size()
            .expect("address_size must be known")
            .bytes();
        self.object.append_section_data(
            init_array_section,
            &std::iter::repeat(0)
                .take(address_size.into())
                .collect::<Vec<u8>>(),
            8,
        );
        self.object
            .add_relocation(
                init_array_section,
                object::write::Relocation {
                    offset: 0,
                    size: address_size * 8,
                    kind: RelocationKind::Absolute,
                    encoding: RelocationEncoding::Generic,
                    symbol,
                    addend: 0,
                },
            )
            .unwrap();
    }
}

pub(crate) fn with_object(sess: &Session, name: &str, f: impl FnOnce(&mut Object)) -> Vec<u8> {
    let triple = crate::build_isa(sess).triple().clone();

    let binary_format = match triple.binary_format {
        target_lexicon::BinaryFormat::Elf => object::BinaryFormat::Elf,
        target_lexicon::BinaryFormat::Coff => object::BinaryFormat::Coff,
        target_lexicon::BinaryFormat::Macho => object::BinaryFormat::MachO,
        binary_format => sess.fatal(&format!("binary format {} is unsupported", binary_format)),
    };
    let architecture = match triple.architecture {
        target_lexicon::Architecture::X86_32(_) => object::Architecture::I386,
        target_lexicon::Architecture::X86_64 => object::Architecture::X86_64,
        target_lexicon::Architecture::Arm(_) => object::Architecture::Arm,
        target_lexicon::Architecture::Aarch64(_) => object::Architecture::Aarch64,
        architecture => sess.fatal(&format!(
            "target architecture {:?} is unsupported",
            architecture,
        )),
    };
    let endian = match triple.endianness().unwrap() {
        target_lexicon::Endianness::Little => object::Endianness::Little,
        target_lexicon::Endianness::Big => object::Endianness::Big,
    };

    let mut metadata_object = object::write::Object::new(binary_format, architecture, endian);
    metadata_object.add_file_symbol(name.as_bytes().to_vec());
    f(&mut metadata_object);
    metadata_object.write().unwrap()
}

pub(crate) fn make_module(sess: &Session, name: String) -> ObjectModule {
    let mut builder = ObjectBuilder::new(
        crate::build_isa(sess),
        name + ".o",
        cranelift_module::default_libcall_names(),
    )
    .unwrap();
    // Unlike cg_llvm, cg_clif defaults to disabling -Zfunction-sections. For cg_llvm binary size
    // is important, while cg_clif cares more about compilation times. Enabling -Zfunction-sections
    // can easily double the amount of time necessary to perform linking.
    builder.per_function_section(sess.opts.debugging_opts.function_sections.unwrap_or(false));
    ObjectModule::new(builder)
}
