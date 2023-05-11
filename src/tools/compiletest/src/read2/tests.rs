use crate::read2::{ProcOutput, FILTERED_PATHS_PLACEHOLDER_LEN, HEAD_LEN, TAIL_LEN};

#[test]
fn test_abbreviate_short_string() {
    let mut out = ProcOutput::new();
    out.extend(b"Hello world!", &[]);
    assert_eq!(b"Hello world!", &*out.into_bytes());
}

#[test]
fn test_abbreviate_short_string_multiple_steps() {
    let mut out = ProcOutput::new();

    out.extend(b"Hello ", &[]);
    out.extend(b"world!", &[]);

    assert_eq!(b"Hello world!", &*out.into_bytes());
}

#[test]
fn test_abbreviate_long_string() {
    let mut out = ProcOutput::new();

    let data = vec![b'.'; HEAD_LEN + TAIL_LEN + 16];
    out.extend(&data, &[]);

    let mut expected = vec![b'.'; HEAD_LEN];
    expected.extend_from_slice(b"\n\n<<<<<< SKIPPED 16 BYTES >>>>>>\n\n");
    expected.extend_from_slice(&vec![b'.'; TAIL_LEN]);

    // We first check the length to avoid endless terminal output if the length differs, since
    // `out` is hundreds of KBs in size.
    let out = out.into_bytes();
    assert_eq!(expected.len(), out.len());
    assert_eq!(expected, out);
}

#[test]
fn test_abbreviate_long_string_multiple_steps() {
    let mut out = ProcOutput::new();

    out.extend(&vec![b'.'; HEAD_LEN], &[]);
    out.extend(&vec![b'.'; TAIL_LEN], &[]);
    // Also test whether the rotation works
    out.extend(&vec![b'!'; 16], &[]);
    out.extend(&vec![b'?'; 16], &[]);

    let mut expected = vec![b'.'; HEAD_LEN];
    expected.extend_from_slice(b"\n\n<<<<<< SKIPPED 32 BYTES >>>>>>\n\n");
    expected.extend_from_slice(&vec![b'.'; TAIL_LEN - 32]);
    expected.extend_from_slice(&vec![b'!'; 16]);
    expected.extend_from_slice(&vec![b'?'; 16]);

    // We first check the length to avoid endless terminal output if the length differs, since
    // `out` is hundreds of KBs in size.
    let out = out.into_bytes();
    assert_eq!(expected.len(), out.len());
    assert_eq!(expected, out);
}

#[test]
fn test_abbreviate_filterss_are_detected() {
    let mut out = ProcOutput::new();
    let filters = &["foo".to_string(), "quux".to_string()];

    out.extend(b"Hello foo", filters);
    // Check items from a previous extension are not double-counted.
    out.extend(b"! This is a qu", filters);
    // Check items are detected across extensions.
    out.extend(b"ux.", filters);

    match &out {
        ProcOutput::Full { bytes, filtered_len } => assert_eq!(
            *filtered_len,
            bytes.len() + FILTERED_PATHS_PLACEHOLDER_LEN * filters.len()
                - filters.iter().map(|i| i.len()).sum::<usize>()
        ),
        ProcOutput::Abbreviated { .. } => panic!("out should not be abbreviated"),
    }

    assert_eq!(b"Hello foo! This is a quux.", &*out.into_bytes());
}

#[test]
fn test_abbreviate_filters_avoid_abbreviations() {
    let mut out = ProcOutput::new();
    let filters = &[std::iter::repeat('a').take(64).collect::<String>()];

    let mut expected = vec![b'.'; HEAD_LEN - FILTERED_PATHS_PLACEHOLDER_LEN as usize];
    expected.extend_from_slice(filters[0].as_bytes());
    expected.extend_from_slice(&vec![b'.'; TAIL_LEN]);

    out.extend(&expected, filters);

    // We first check the length to avoid endless terminal output if the length differs, since
    // `out` is hundreds of KBs in size.
    let out = out.into_bytes();
    assert_eq!(expected.len(), out.len());
    assert_eq!(expected, out);
}

#[test]
fn test_abbreviate_filters_can_still_cause_abbreviations() {
    let mut out = ProcOutput::new();
    let filters = &[std::iter::repeat('a').take(64).collect::<String>()];

    let mut input = vec![b'.'; HEAD_LEN];
    input.extend_from_slice(&vec![b'.'; TAIL_LEN]);
    input.extend_from_slice(filters[0].as_bytes());

    let mut expected = vec![b'.'; HEAD_LEN];
    expected.extend_from_slice(b"\n\n<<<<<< SKIPPED 64 BYTES >>>>>>\n\n");
    expected.extend_from_slice(&vec![b'.'; TAIL_LEN - 64]);
    expected.extend_from_slice(&vec![b'a'; 64]);

    out.extend(&input, filters);

    // We first check the length to avoid endless terminal output if the length differs, since
    // `out` is hundreds of KBs in size.
    let out = out.into_bytes();
    assert_eq!(expected.len(), out.len());
    assert_eq!(expected, out);
}
