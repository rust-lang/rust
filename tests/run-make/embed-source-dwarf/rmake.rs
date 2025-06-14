//@ needs-target-std
//@ ignore-windows
//@ ignore-apple

// This test should be replaced with one in tests/debuginfo once we can easily
// tell via GDB or LLDB if debuginfo contains source code. Cheap tricks in LLDB
// like setting an invalid source map path don't appear to work, maybe this'll
// become easier once GDB supports DWARFv6?

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use gimli::{EndianRcSlice, Reader, RunTimeEndian};
use object::{Object, ObjectSection};
use run_make_support::{gimli, object, rfs, rustc};

fn main() {
    let output = PathBuf::from("embed-source-main");
    rustc()
        .input("main.rs")
        .output(&output)
        .arg("-g")
        .arg("-Zembed-source=yes")
        .arg("-Cdwarf-version=5")
        .run();
    let output = rfs::read(output);
    let obj = object::File::parse(output.as_slice()).unwrap();
    let endian = if obj.is_little_endian() { RunTimeEndian::Little } else { RunTimeEndian::Big };
    let dwarf = gimli::Dwarf::load(|section| -> Result<_, ()> {
        let data = obj.section_by_name(section.name()).map(|s| s.uncompressed_data().unwrap());
        Ok(EndianRcSlice::new(Rc::from(data.unwrap_or_default().as_ref()), endian))
    })
    .unwrap();

    let mut sources = HashMap::new();

    let mut iter = dwarf.units();
    while let Some(header) = iter.next().unwrap() {
        let unit = dwarf.unit(header).unwrap();
        let unit = unit.unit_ref(&dwarf);

        if let Some(program) = &unit.line_program {
            let header = program.header();
            for file in header.file_names() {
                if let Some(source) = file.source() {
                    let path = unit
                        .attr_string(file.path_name())
                        .unwrap()
                        .to_string_lossy()
                        .unwrap()
                        .to_string();
                    let source =
                        unit.attr_string(source).unwrap().to_string_lossy().unwrap().to_string();
                    if !source.is_empty() {
                        sources.insert(path, source);
                    }
                }
            }
        }
    }

    dbg!(&sources);
    assert_eq!(sources.len(), 1);
    assert_eq!(sources.get("main.rs").unwrap(), "// hello\nfn main() {}\n");
}
