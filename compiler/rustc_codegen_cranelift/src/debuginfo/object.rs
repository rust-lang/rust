use cranelift_module::{DataId, FuncId};
use cranelift_object::ObjectProduct;
use gimli::SectionId;
use object::write::{Relocation, StandardSegment};
use object::{RelocationEncoding, RelocationFlags, SectionKind};
use rustc_data_structures::fx::FxHashMap;

use crate::debuginfo::{DebugReloc, DebugRelocName};

pub(super) trait WriteDebugInfo {
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
            if id == SectionId::DebugStr || id == SectionId::DebugLineStr {
                SectionKind::DebugString
            } else if id == SectionId::EhFrame {
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
                let id = id.try_into().unwrap();
                let symbol_id = if id & 1 << 31 == 0 {
                    self.function_symbol(FuncId::from_u32(id))
                } else {
                    self.data_symbol(DataId::from_u32(id & !(1 << 31)))
                };
                self.object.symbol_section_and_offset(symbol_id).unwrap_or((symbol_id, 0))
            }
        };
        self.object
            .add_relocation(
                from.0,
                Relocation {
                    offset: u64::from(reloc.offset),
                    symbol,
                    flags: RelocationFlags::Generic {
                        kind: reloc.kind,
                        encoding: RelocationEncoding::Generic,
                        size: reloc.size * 8,
                    },
                    addend: i64::try_from(symbol_offset).unwrap() + reloc.addend,
                },
            )
            .unwrap();
    }
}
