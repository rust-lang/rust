//! Write the debuginfo into an object file.

use cranelift_module::{DataId, FuncId};
use cranelift_object::ObjectProduct;
use gimli::write::{Address, AttributeValue, EndianVec, Result, Sections, Writer};
use gimli::{RunTimeEndian, SectionId};
use rustc_data_structures::fx::FxHashMap;

use super::DebugContext;
use super::object::WriteDebugInfo;

pub(super) fn address_for_func(func_id: FuncId) -> Address {
    let symbol = func_id.as_u32();
    assert!(symbol & 1 << 31 == 0);
    Address::Symbol { symbol: symbol as usize, addend: 0 }
}

pub(super) fn address_for_data(data_id: DataId) -> Address {
    let symbol = data_id.as_u32();
    assert!(symbol & 1 << 31 == 0);
    Address::Symbol { symbol: (symbol | 1 << 31) as usize, addend: 0 }
}

impl DebugContext {
    pub(crate) fn emit(&mut self, product: &mut ObjectProduct) {
        let unit_range_list_id = self.dwarf.unit.ranges.add(self.unit_range_list.clone());
        let root = self.dwarf.unit.root();
        let root = self.dwarf.unit.get_mut(root);
        root.set(gimli::DW_AT_ranges, AttributeValue::RangeListRef(unit_range_list_id));

        let mut sections = Sections::new(WriterRelocate::new(self.endian));
        self.dwarf.write(&mut sections).unwrap();

        let mut section_map = FxHashMap::default();
        let _: Result<()> = sections.for_each_mut(|id, section| {
            if !section.writer.slice().is_empty() {
                let section_id = product.add_debug_section(id, section.writer.take());
                section_map.insert(id, section_id);
            }
            Ok(())
        });

        let _: Result<()> = sections.for_each(|id, section| {
            if let Some(section_id) = section_map.get(&id) {
                for reloc in &section.relocs {
                    product.add_debug_reloc(&section_map, section_id, reloc);
                }
            }
            Ok(())
        });
    }
}

#[derive(Clone)]
pub(crate) struct DebugReloc {
    pub(crate) offset: u32,
    pub(crate) size: u8,
    pub(crate) name: DebugRelocName,
    pub(crate) addend: i64,
    pub(crate) kind: object::RelocationKind,
}

#[derive(Clone)]
pub(crate) enum DebugRelocName {
    Section(SectionId),
    Symbol(usize),
}

/// A [`Writer`] that collects all necessary relocations.
#[derive(Clone)]
pub(super) struct WriterRelocate {
    pub(super) relocs: Vec<DebugReloc>,
    pub(super) writer: EndianVec<RunTimeEndian>,
}

impl WriterRelocate {
    pub(super) fn new(endian: RunTimeEndian) -> Self {
        WriterRelocate { relocs: Vec::new(), writer: EndianVec::new(endian) }
    }

    /// Perform the collected relocations to be usable for JIT usage.
    #[cfg(all(feature = "jit", not(windows)))]
    pub(super) fn relocate_for_jit(mut self, jit_module: &cranelift_jit::JITModule) -> Vec<u8> {
        use cranelift_module::Module;

        for reloc in self.relocs.drain(..) {
            match reloc.name {
                super::DebugRelocName::Section(_) => unreachable!(),
                super::DebugRelocName::Symbol(sym) => {
                    let addr = if sym & 1 << 31 == 0 {
                        let func_id = FuncId::from_u32(sym.try_into().unwrap());
                        // FIXME make JITModule::get_address public and use it here instead.
                        // HACK rust_eh_personality is likely not defined in the same crate,
                        // so get_finalized_function won't work. Use the rust_eh_personality
                        // of cg_clif itself, which is likely ABI compatible.
                        if jit_module.declarations().get_function_decl(func_id).name.as_deref()
                            == Some("rust_eh_personality")
                        {
                            extern "C" {
                                fn rust_eh_personality() -> !;
                            }
                            rust_eh_personality as *const u8
                        } else {
                            jit_module.get_finalized_function(func_id)
                        }
                    } else {
                        jit_module
                            .get_finalized_data(DataId::from_u32(
                                u32::try_from(sym).unwrap() & !(1 << 31),
                            ))
                            .0
                    };

                    let val = (addr as u64 as i64 + reloc.addend) as u64;
                    self.writer.write_udata_at(reloc.offset as usize, val, reloc.size).unwrap();
                }
            }
        }
        self.writer.into_vec()
    }
}

impl Writer for WriterRelocate {
    type Endian = RunTimeEndian;

    fn endian(&self) -> Self::Endian {
        self.writer.endian()
    }

    fn len(&self) -> usize {
        self.writer.len()
    }

    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write(bytes)
    }

    fn write_at(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.writer.write_at(offset, bytes)
    }

    fn write_address(&mut self, address: Address, size: u8) -> Result<()> {
        match address {
            Address::Constant(val) => self.write_udata(val, size),
            Address::Symbol { symbol, addend } => {
                let offset = self.len() as u64;
                self.relocs.push(DebugReloc {
                    offset: offset as u32,
                    size,
                    name: DebugRelocName::Symbol(symbol),
                    addend,
                    kind: object::RelocationKind::Absolute,
                });
                self.write_udata(0, size)
            }
        }
    }

    fn write_offset(&mut self, val: usize, section: SectionId, size: u8) -> Result<()> {
        let offset = self.len() as u32;
        self.relocs.push(DebugReloc {
            offset,
            size,
            name: DebugRelocName::Section(section),
            addend: val as i64,
            kind: object::RelocationKind::Absolute,
        });
        self.write_udata(0, size)
    }

    fn write_offset_at(
        &mut self,
        offset: usize,
        val: usize,
        section: SectionId,
        size: u8,
    ) -> Result<()> {
        self.relocs.push(DebugReloc {
            offset: offset as u32,
            size,
            name: DebugRelocName::Section(section),
            addend: val as i64,
            kind: object::RelocationKind::Absolute,
        });
        self.write_udata_at(offset, 0, size)
    }

    fn write_eh_pointer(&mut self, address: Address, eh_pe: gimli::DwEhPe, size: u8) -> Result<()> {
        match address {
            // Address::Constant arm copied from gimli
            Address::Constant(val) => {
                // Indirect doesn't matter here.
                let val = match eh_pe.application() {
                    gimli::DW_EH_PE_absptr => val,
                    gimli::DW_EH_PE_pcrel => {
                        // FIXME better handling of sign
                        let offset = self.len() as u64;
                        offset.wrapping_sub(val)
                    }
                    _ => {
                        return Err(gimli::write::Error::UnsupportedPointerEncoding(eh_pe));
                    }
                };
                self.write_eh_pointer_data(val, eh_pe.format(), size)
            }
            Address::Symbol { symbol, addend } => match eh_pe.application() {
                gimli::DW_EH_PE_pcrel => {
                    let size = match eh_pe.format() {
                        gimli::DW_EH_PE_sdata4 => 4,
                        gimli::DW_EH_PE_sdata8 => 8,
                        _ => return Err(gimli::write::Error::UnsupportedPointerEncoding(eh_pe)),
                    };
                    self.relocs.push(DebugReloc {
                        offset: self.len() as u32,
                        size,
                        name: DebugRelocName::Symbol(symbol),
                        addend,
                        kind: object::RelocationKind::Relative,
                    });
                    self.write_udata(0, size)
                }
                gimli::DW_EH_PE_absptr => {
                    self.relocs.push(DebugReloc {
                        offset: self.len() as u32,
                        size: size.into(),
                        name: DebugRelocName::Symbol(symbol),
                        addend,
                        kind: object::RelocationKind::Absolute,
                    });
                    self.write_udata(0, size.into())
                }
                _ => Err(gimli::write::Error::UnsupportedPointerEncoding(eh_pe)),
            },
        }
    }
}
