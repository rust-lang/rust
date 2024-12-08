//@ ignore-windows
// This test should be replaced with one in tests/debuginfo once GDB or LLDB support 128-bit enums.

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use gimli::{AttributeValue, EndianRcSlice, Reader, RunTimeEndian};
use object::{Object, ObjectSection};
use run_make_support::{gimli, object, rfs, rustc};

fn main() {
    let output = PathBuf::from("repr128");
    rustc().input("main.rs").output(&output).arg("-Cdebuginfo=2").run();
    // Mach-O uses packed debug info
    let dsym_location = output
        .with_extension("dSYM")
        .join("Contents")
        .join("Resources")
        .join("DWARF")
        .join("repr128");
    let output =
        rfs::read(if dsym_location.try_exists().unwrap() { dsym_location } else { output });
    let obj = object::File::parse(output.as_slice()).unwrap();
    let endian = if obj.is_little_endian() { RunTimeEndian::Little } else { RunTimeEndian::Big };
    let dwarf = gimli::Dwarf::load(|section| -> Result<_, ()> {
        let data = obj.section_by_name(section.name()).map(|s| s.uncompressed_data().unwrap());
        Ok(EndianRcSlice::new(Rc::from(data.unwrap_or_default().as_ref()), endian))
    })
    .unwrap();
    let mut iter = dwarf.units();
    let mut still_to_find = HashMap::from([
        ("U128A", 0_u128),
        ("U128B", 1_u128),
        ("U128C", u64::MAX as u128 + 1),
        ("U128D", u128::MAX),
        ("I128A", 0_i128 as u128),
        ("I128B", (-1_i128) as u128),
        ("I128C", i128::MIN as u128),
        ("I128D", i128::MAX as u128),
    ]);
    while let Some(header) = iter.next().unwrap() {
        let unit = dwarf.unit(header).unwrap();
        let mut cursor = unit.entries();
        while let Some((_, entry)) = cursor.next_dfs().unwrap() {
            if entry.tag() == gimli::constants::DW_TAG_enumerator {
                let name = dwarf
                    .attr_string(
                        &unit,
                        entry.attr(gimli::constants::DW_AT_name).unwrap().unwrap().value(),
                    )
                    .unwrap();
                let name = name.to_string().unwrap();
                if let Some(expected) = still_to_find.remove(name.as_ref()) {
                    match entry.attr(gimli::constants::DW_AT_const_value).unwrap().unwrap().value()
                    {
                        AttributeValue::Block(value) => {
                            assert_eq!(
                                value.to_slice().unwrap(),
                                expected.to_le_bytes().as_slice(),
                                "{name}"
                            );
                        }
                        value => panic!("{name}: unexpected DW_AT_const_value of {value:?}"),
                    }
                }
            }
        }
    }
    if !still_to_find.is_empty() {
        panic!("Didn't find debug entries for {still_to_find:?}");
    }
}
