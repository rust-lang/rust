// basic examples
#[test]
fn escape_body_text_with_wbr() {
    use super::EscapeBodyTextWithWbr as E;
    // extreme corner cases
    assert_eq!(&E("").to_string(), "");
    assert_eq!(&E("a").to_string(), "a");
    assert_eq!(&E("A").to_string(), "A");
    assert_eq!(&E("_").to_string(), "_");
    assert_eq!(&E(":").to_string(), ":");
    assert_eq!(&E(" ").to_string(), " ");
    assert_eq!(&E("___________").to_string(), "___________");
    assert_eq!(&E(":::::::::::").to_string(), ":::::::::::");
    assert_eq!(&E("           ").to_string(), "           ");
    // real(istic) examples
    assert_eq!(&E("FirstSecond").to_string(), "First<wbr>Second");
    assert_eq!(&E("First_Second").to_string(), "First_<wbr>Second");
    assert_eq!(&E("First Second").to_string(), "First Second");
    assert_eq!(&E("First HSecond").to_string(), "First HSecond");
    assert_eq!(&E("First HTTPSecond").to_string(), "First HTTP<wbr>Second");
    assert_eq!(&E("First SecondThird").to_string(), "First Second<wbr>Third");
    assert_eq!(&E("First<T>_Second").to_string(), "First&lt;<wbr>T&gt;_<wbr>Second");
    assert_eq!(&E("first_second").to_string(), "first_<wbr>second");
    assert_eq!(&E("first:second").to_string(), "first:<wbr>second");
    assert_eq!(&E("first::second").to_string(), "first::<wbr>second");
    assert_eq!(&E("MY_CONSTANT").to_string(), "MY_<wbr>CONSTANT");
    assert_eq!(
        &E("_SIDD_MASKED_NEGATIVE_POLARITY").to_string(),
        "_SIDD_<wbr>MASKED_<wbr>NEGATIVE_<wbr>POLARITY"
    );
    // a string won't get wrapped if it's less than 8 bytes
    assert_eq!(&E("HashSet").to_string(), "HashSet");
    // an individual word won't get wrapped if it's less than 4 bytes
    assert_eq!(&E("VecDequeue").to_string(), "VecDequeue");
    assert_eq!(&E("VecDequeueSet").to_string(), "VecDequeue<wbr>Set");
    // how to handle acronyms
    assert_eq!(&E("BTreeMap").to_string(), "BTree<wbr>Map");
    assert_eq!(&E("HTTPSProxy").to_string(), "HTTPS<wbr>Proxy");
    // more corners
    assert_eq!(&E("ṼẽçÑñéå").to_string(), "Ṽẽç<wbr>Ññéå");
    assert_eq!(&E("V\u{0300}e\u{0300}c\u{0300}D\u{0300}e\u{0300}q\u{0300}u\u{0300}e\u{0300}u\u{0300}e\u{0300}").to_string(), "V\u{0300}e\u{0300}c\u{0300}<wbr>D\u{0300}e\u{0300}q\u{0300}u\u{0300}e\u{0300}u\u{0300}e\u{0300}");
    assert_eq!(&E("LPFNACCESSIBLEOBJECTFROMWINDOW").to_string(), "LPFNACCESSIBLEOBJECTFROMWINDOW");
}
// property test
#[test]
fn escape_body_text_with_wbr_makes_sense() {
    use itertools::Itertools as _;

    use super::EscapeBodyTextWithWbr as E;
    const C: [u8; 3] = [b'a', b'A', b'_'];
    for chars in [
        C.into_iter(),
        C.into_iter(),
        C.into_iter(),
        C.into_iter(),
        C.into_iter(),
        C.into_iter(),
        C.into_iter(),
        C.into_iter(),
    ]
    .into_iter()
    .multi_cartesian_product()
    {
        let s = String::from_utf8(chars).unwrap();
        assert_eq!(s.len(), 8);
        let esc = E(&s).to_string();
        assert!(!esc.contains("<wbr><wbr>"));
        assert!(!esc.ends_with("<wbr>"));
        assert!(!esc.starts_with("<wbr>"));
        assert_eq!(&esc.replace("<wbr>", ""), &s);
    }
}
