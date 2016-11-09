extern crate toml;

use toml::{Parser};

fn run(toml: &str) {
    let mut p = Parser::new(toml);
    let table = p.parse();
    assert!(table.is_none());
    assert!(p.errors.len() > 0);

    // test Parser::to_linecol with the generated error offsets
    for error in &p.errors {
      p.to_linecol(error.lo);
      p.to_linecol(error.hi);
    }
}

macro_rules! test( ($name:ident, $toml:expr) => (
    #[test]
    fn $name() { run($toml); }
) );

test!(array_mixed_types_arrays_and_ints,
      include_str!("invalid/array-mixed-types-arrays-and-ints.toml"));
test!(array_mixed_types_ints_and_floats,
      include_str!("invalid/array-mixed-types-ints-and-floats.toml"));
test!(array_mixed_types_strings_and_ints,
      include_str!("invalid/array-mixed-types-strings-and-ints.toml"));
test!(datetime_malformed_no_leads,
      include_str!("invalid/datetime-malformed-no-leads.toml"));
test!(datetime_malformed_no_secs,
      include_str!("invalid/datetime-malformed-no-secs.toml"));
test!(datetime_malformed_no_t,
      include_str!("invalid/datetime-malformed-no-t.toml"));
test!(datetime_malformed_no_z,
      include_str!("invalid/datetime-malformed-no-z.toml"));
test!(datetime_malformed_with_milli,
      include_str!("invalid/datetime-malformed-with-milli.toml"));
test!(duplicate_keys,
      include_str!("invalid/duplicate-keys.toml"));
test!(duplicate_key_table,
      include_str!("invalid/duplicate-key-table.toml"));
test!(duplicate_tables,
      include_str!("invalid/duplicate-tables.toml"));
test!(empty_implicit_table,
      include_str!("invalid/empty-implicit-table.toml"));
test!(empty_table,
      include_str!("invalid/empty-table.toml"));
test!(float_no_leading_zero,
      include_str!("invalid/float-no-leading-zero.toml"));
test!(float_no_trailing_digits,
      include_str!("invalid/float-no-trailing-digits.toml"));
test!(key_after_array,
      include_str!("invalid/key-after-array.toml"));
test!(key_after_table,
      include_str!("invalid/key-after-table.toml"));
test!(key_empty,
      include_str!("invalid/key-empty.toml"));
test!(key_hash,
      include_str!("invalid/key-hash.toml"));
test!(key_newline,
      include_str!("invalid/key-newline.toml"));
test!(key_open_bracket,
      include_str!("invalid/key-open-bracket.toml"));
test!(key_single_open_bracket,
      include_str!("invalid/key-single-open-bracket.toml"));
test!(key_space,
      include_str!("invalid/key-space.toml"));
test!(key_start_bracket,
      include_str!("invalid/key-start-bracket.toml"));
test!(key_two_equals,
      include_str!("invalid/key-two-equals.toml"));
test!(string_bad_byte_escape,
      include_str!("invalid/string-bad-byte-escape.toml"));
test!(string_bad_escape,
      include_str!("invalid/string-bad-escape.toml"));
test!(string_byte_escapes,
      include_str!("invalid/string-byte-escapes.toml"));
test!(string_no_close,
      include_str!("invalid/string-no-close.toml"));
test!(table_array_implicit,
      include_str!("invalid/table-array-implicit.toml"));
test!(table_array_malformed_bracket,
      include_str!("invalid/table-array-malformed-bracket.toml"));
test!(table_array_malformed_empty,
      include_str!("invalid/table-array-malformed-empty.toml"));
test!(table_empty,
      include_str!("invalid/table-empty.toml"));
test!(table_nested_brackets_close,
      include_str!("invalid/table-nested-brackets-close.toml"));
test!(table_nested_brackets_open,
      include_str!("invalid/table-nested-brackets-open.toml"));
test!(table_whitespace,
      include_str!("invalid/table-whitespace.toml"));
test!(table_with_pound,
      include_str!("invalid/table-with-pound.toml"));
test!(text_after_array_entries,
      include_str!("invalid/text-after-array-entries.toml"));
test!(text_after_integer,
      include_str!("invalid/text-after-integer.toml"));
test!(text_after_string,
      include_str!("invalid/text-after-string.toml"));
test!(text_after_table,
      include_str!("invalid/text-after-table.toml"));
test!(text_before_array_separator,
      include_str!("invalid/text-before-array-separator.toml"));
test!(text_in_array,
      include_str!("invalid/text-in-array.toml"));
