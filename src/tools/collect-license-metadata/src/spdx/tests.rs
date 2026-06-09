use super::*;

// Clause 8.1 ("File name field") specifies that each file section begins with
// a `FileName` tag whose value is a relative path prefixed with "./".
// Clause 8.5 ("Concluded license") and 8.8 ("Copyright text") give the
// corresponding per-file fields.
// <https://spdx.github.io/spdx-spec/v2.3/file-information/>
#[test]
fn single_file_entry() {
    let input = "\
FileName: ./package/foo.c
LicenseConcluded: LGPL-2.0-only
FileCopyrightText: Copyright 2008-2010 John Smith";

    let files = parse_tag_value(input).unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].name, "./package/foo.c");
    assert_eq!(files[0].concluded_license, "LGPL-2.0-only");
    assert_eq!(files[0].copyright_text, "Copyright 2008-2010 John Smith");
}

// Clause 8.5 shows compound SPDX licence expressions as valid values for
// `LicenseConcluded`, e.g. "(LGPL-2.0-only OR LicenseRef-2)".
// <https://spdx.github.io/spdx-spec/v2.3/file-information/>
#[test]
fn compound_license_expression() {
    let input = "\
FileName: ./src/lib.rs
LicenseConcluded: (LGPL-2.0-only OR LicenseRef-2)
FileCopyrightText: Copyright Example Company";

    let files = parse_tag_value(input).unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].concluded_license, "(LGPL-2.0-only OR LicenseRef-2)");
}

// Clause 8.8 shows the copyright text wrapped in a single-line
// <text>...</text> block: e.g.
// `FileCopyrightText: <text>Copyright 2008-2010 John Smith</text>`
// <https://spdx.github.io/spdx-spec/v2.3/file-information/>
#[test]
fn single_line_text_block() {
    let input = "\
FileName: ./package/foo.c
LicenseConcluded: LGPL-2.0-only
FileCopyrightText: <text>Copyright 2008-2010 John Smith</text>";

    let files = parse_tag_value(input).unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].copyright_text, "Copyright 2008-2010 John Smith");
}

// Clause 6.10 ("Creator comment") demonstrates a multi-line <text>...</text> block.
// <https://spdx.github.io/spdx-spec/v2.3/document-creation-information/>
#[test]
fn multi_line_text_block() {
    let input = "\
FileName: ./package/foo.c
LicenseConcluded: MIT
FileCopyrightText: <text>Copyright 2008-2010 John Smith
Copyright 2019 Jane Doe</text>";

    let files = parse_tag_value(input).unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].copyright_text, "Copyright 2008-2010 John Smith\nCopyright 2019 Jane Doe");
}

// Clause 5 ("Composition of an SPDX document") states that a document may
// contain zero or many File Information sections. Each `FileName` tag starts
// a new section, so consecutive file blocks must be parsed independently.
// <https://spdx.github.io/spdx-spec/v2.3/composition-of-an-SPDX-document/>
#[test]
fn multiple_file_entries() {
    let input = "\
FileName: ./package/foo.c
LicenseConcluded: LGPL-2.0-only
FileCopyrightText: Copyright 2008-2010 John Smith
FileName: ./package/bar.c
LicenseConcluded: MIT
FileCopyrightText: Copyright Example Company";

    let files = parse_tag_value(input).unwrap();
    assert_eq!(files.len(), 2);

    assert_eq!(files[0].name, "./package/foo.c");
    assert_eq!(files[0].concluded_license, "LGPL-2.0-only");
    assert_eq!(files[0].copyright_text, "Copyright 2008-2010 John Smith");

    assert_eq!(files[1].name, "./package/bar.c");
    assert_eq!(files[1].concluded_license, "MIT");
    assert_eq!(files[1].copyright_text, "Copyright Example Company");
}

// A file section without a `LicenseConcluded` tag is malformed.
#[test]
fn missing_license_is_an_error() {
    let input = "\
FileName: ./package/foo.c
FileCopyrightText: Copyright 2008-2010 John Smith";

    assert!(parse_tag_value(input).is_err());
}

// A file section without a `FileCopyrightText` tag is malformed.
#[test]
fn missing_copyright_is_an_error() {
    let input = "\
FileName: ./package/foo.c
LicenseConcluded: MIT";

    assert!(parse_tag_value(input).is_err());
}

// A section with an unterminated <text> block (no closing </text>) is malformed.
#[test]
fn unterminated_text_block_is_an_error() {
    let input = "\
FileName: ./package/foo.c
LicenseConcluded: MIT
FileCopyrightText: <text>Copyright 2008-2010 John Smith";

    assert!(parse_tag_value(input).is_err());
}

// A document with no `FileName` tags at all should produce an empty result.
#[test]
fn empty_document_returns_no_entries() {
    let input = "\
SPDXVersion: SPDX-2.3
DataLicense: CC0-1.0";

    let files = parse_tag_value(input).unwrap();
    assert!(files.is_empty());
}
