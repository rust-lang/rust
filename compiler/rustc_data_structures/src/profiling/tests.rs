use super::JsonTimePassesEntry;

#[test]
fn with_rss() {
    let entry =
        JsonTimePassesEntry { pass: "typeck", time: 56.1, start_rss: Some(10), end_rss: Some(20) };

    assert_eq!(entry.to_string(), r#"{"pass":"typeck","time":56.1,"rss_start":10,"rss_end":20}"#)
}

#[test]
fn no_rss() {
    let entry = JsonTimePassesEntry { pass: "typeck", time: 56.1, start_rss: None, end_rss: None };

    assert_eq!(
        entry.to_string(),
        r#"{"pass":"typeck","time":56.1,"rss_start":null,"rss_end":null}"#
    )
}

/// The reader must see the resident set grow when the process touches new memory,
/// and must agree with `get_resident_set_size`, which reads the same file.
///
/// This is one test, not two, because tests in this process run in parallel. A
/// second test that frees its own ballast between two samples of this one would
/// hide the growth.
#[cfg(target_os = "linux")]
#[test]
fn rss_reader() {
    use super::{RssReader, get_resident_set_size};

    const MIB: u64 = 1 << 20;
    const BALLAST: u64 = 64 * MIB;
    // Other tests allocate and free memory between the samples. The ballast makes
    // the expected values large against that noise.
    const SLACK: u64 = 16 * MIB;

    let reader = RssReader::open().expect("/proc/self/statm must open on Linux");
    let empty = reader.read().expect("read before the ballast");

    // Write to each page, because zeroed pages are not resident until written.
    let mut ballast = vec![0u8; BALLAST as usize];
    for byte in ballast.iter_mut().step_by(4096) {
        *byte = 1;
    }
    let ballast = std::hint::black_box(ballast);

    let before = reader.read().expect("first read with the ballast");
    let reference = get_resident_set_size().expect("reference read") as u64;
    // A second read fails if the reader depends on the file position.
    let after = reader.read().expect("second read with the ballast");
    drop(ballast);

    for sample in [empty, before, after] {
        assert_eq!(sample % reader.page_size, 0, "{sample} is not a whole number of pages");
    }

    assert!(
        before >= empty + BALLAST - SLACK,
        "resident set went from {empty} to {before} after {BALLAST} bytes were written"
    );

    let low = before.min(after).saturating_sub(SLACK);
    let high = before.max(after) + SLACK;
    assert!(
        (low..=high).contains(&reference),
        "get_resident_set_size returned {reference}, reader returned {before} and {after}"
    );
}
