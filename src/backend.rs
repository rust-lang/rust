pub trait Product {
    fn add_rustc_section(&mut self, symbol_name: String, data: Vec<u8>, is_like_osx: bool);
}

impl Product for faerie::Artifact {
    fn add_rustc_section(&mut self, symbol_name: String, data: Vec<u8>, is_like_osx: bool) {
        self
            .declare(".rustc", faerie::Decl::section(faerie::SectionKind::Data))
            .unwrap();
        self
            .define_with_symbols(".rustc", data, {
                let mut map = std::collections::BTreeMap::new();
                // FIXME implement faerie elf backend section custom symbols
                // For MachO this is necessary to prevent the linker from throwing away the .rustc section,
                // but for ELF it isn't.
                if is_like_osx {
                    map.insert(
                        symbol_name,
                        0,
                    );
                }
                map
            })
            .unwrap();
    }
}

impl Product for object::write::Object {
    fn add_rustc_section(&mut self, symbol_name: String, data: Vec<u8>, is_like_osx: bool) {
        let segment = self.segment_name(object::write::StandardSegment::Data).to_vec();
        let section_id = self.add_section(segment, b".rustc".to_vec(), object::SectionKind::Data);
        let offset = self.append_section_data(section_id, &data, 1);
        // FIXME implement faerie elf backend section custom symbols
        // For MachO this is necessary to prevent the linker from throwing away the .rustc section,
        // but for ELF it isn't.
        if is_like_osx {
            self.add_symbol(object::write::Symbol {
                name: symbol_name.into_bytes(),
                value: offset,
                size: data.len() as u64,
                kind: object::SymbolKind::Data,
                scope: object::SymbolScope::Compilation,
                weak: false,
                section: Some(section_id),
            });
        }
    }
}
