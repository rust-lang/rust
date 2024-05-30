use crate::{fs_wrapper, object};
use object::{Object, ObjectSection};
use std::path::Path;

#[derive(Debug)]
pub struct Nm {
    file: Option<object::File>,
}

pub fn nm() -> Nm {
    Nm::new()
}

impl Nm {
    /// Construct a bare `nm` invocation.
    pub fn new() -> Self {
        Self { file: None }
    }

    /// Specify the file to analyze the symbols of.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        &mut Self {
            file: Some(
                object::File::parse(fs_wrapper::read(path))
                    .expect(format!("Failed to parse ELF file at {:?}", path.as_ref().display())),
            ),
        }
    }

    /// Collect all symbols of an object file into a String.
    pub fn collect_symbols(&self) -> String {
        let object_file = self.file;
        let mut symbols_str = String::new();
        for section in object_file.sections() {
            if let Ok(ObjectSection::SymbolTable(st)) = section.parse::<object::SymbolTable>() {
                for symbol in st.symbols() {
                    symbols_str.push_str(&format!(
                        "{:016x} {:?} {}\n",
                        symbol.address(),
                        symbol.kind(),
                        symbol.name()
                    ));
                }
            }
        }
        symbols_str
    }
}
