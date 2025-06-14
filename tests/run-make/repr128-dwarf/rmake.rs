//@ needs-target-std
//@ ignore-windows
// This test should be replaced with one in tests/debuginfo once GDB or LLDB support 128-bit enums.

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use gimli::read::DebuggingInformationEntry;
use gimli::{AttributeValue, EndianRcSlice, Reader, RunTimeEndian};
use object::{Object, ObjectSection};
use run_make_support::{gimli, object, rfs, rustc};

fn main() {
    // Before LLVM 20, 128-bit enums with variants didn't emit debuginfo correctly.
    // This check can be removed once Rust no longer supports LLVM 18 and 19.
    let llvm_version = rustc()
        .verbose()
        .arg("--version")
        .run()
        .stdout_utf8()
        .lines()
        .filter_map(|line| line.strip_prefix("LLVM version: "))
        .map(|version| version.split(".").next().unwrap().parse::<u32>().unwrap())
        .next()
        .unwrap();
    let is_old_llvm = llvm_version < 20;

    let output = PathBuf::from("repr128");
    let mut rustc = rustc();
    if is_old_llvm {
        rustc.cfg("old_llvm");
    }
    rustc.input("main.rs").output(&output).arg("-Cdebuginfo=2").run();
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

    let mut enumerators_to_find = HashMap::from([
        ("U128A", 0_u128),
        ("U128B", 1_u128),
        ("U128C", u64::MAX as u128 + 1),
        ("U128D", u128::MAX),
        ("I128A", 0_i128 as u128),
        ("I128B", (-1_i128) as u128),
        ("I128C", i128::MIN as u128),
        ("I128D", i128::MAX as u128),
    ]);
    let mut variants_to_find = HashMap::from([
        ("VariantU128A", 0_u128),
        ("VariantU128B", 1_u128),
        ("VariantU128C", u64::MAX as u128 + 1),
        ("VariantU128D", u128::MAX),
        ("VariantI128A", 0_i128 as u128),
        ("VariantI128B", (-1_i128) as u128),
        ("VariantI128C", i128::MIN as u128),
        ("VariantI128D", i128::MAX as u128),
    ]);

    while let Some(header) = iter.next().unwrap() {
        let unit = dwarf.unit(header).unwrap();
        let mut cursor = unit.entries();

        let get_name = |entry: &DebuggingInformationEntry<'_, '_, _>| {
            let name = dwarf
                .attr_string(
                    &unit,
                    entry.attr(gimli::constants::DW_AT_name).unwrap().unwrap().value(),
                )
                .unwrap();
            name.to_string().unwrap().to_string()
        };

        while let Some((_, entry)) = cursor.next_dfs().unwrap() {
            match entry.tag() {
                gimli::constants::DW_TAG_variant if !is_old_llvm => {
                    let Some(value) = entry.attr(gimli::constants::DW_AT_discr_value).unwrap()
                    else {
                        // `std` enums might have variants without `DW_AT_discr_value`.
                        continue;
                    };
                    let value = match value.value() {
                        AttributeValue::Block(value) => value.to_slice().unwrap().to_vec(),
                        // `std` has non-repr128 enums which don't use `AttributeValue::Block`.
                        value => continue,
                    };
                    // The `DW_TAG_member` that is a child of `DW_TAG_variant` will contain the
                    // variant's name.
                    let Some((1, child_entry)) = cursor.next_dfs().unwrap() else {
                        panic!("Missing child of DW_TAG_variant");
                    };
                    assert_eq!(child_entry.tag(), gimli::constants::DW_TAG_member);
                    let name = get_name(child_entry);
                    if let Some(expected) = variants_to_find.remove(name.as_str()) {
                        // This test uses LE byte order is used for consistent values across
                        // architectures.
                        assert_eq!(value.as_slice(), expected.to_le_bytes().as_slice(), "{name}");
                    }
                }

                gimli::constants::DW_TAG_enumerator => {
                    let name = get_name(entry);
                    if let Some(expected) = enumerators_to_find.remove(name.as_str()) {
                        match entry
                            .attr(gimli::constants::DW_AT_const_value)
                            .unwrap()
                            .unwrap()
                            .value()
                        {
                            AttributeValue::Block(value) => {
                                // This test uses LE byte order is used for consistent values across
                                // architectures.
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

                _ => {}
            }
        }
    }
    if !enumerators_to_find.is_empty() {
        panic!("Didn't find debug enumerator entries for {enumerators_to_find:?}");
    }
    if !is_old_llvm && !variants_to_find.is_empty() {
        panic!("Didn't find debug variant entries for {variants_to_find:?}");
    }
}
