pub fn main()
{
    let all_nuls1 = "\0\x00\u{0}\u{0}";
    let all_nuls2 = "\u{0}\u{0}\x00\0";
    let all_nuls3 = "\u{0}\u{0}\x00\0";
    let all_nuls4 = "\x00\u{0}\0\u{0}";

    // sizes for two should suffice
    assert_eq!(all_nuls1.len(), 4);
    assert_eq!(all_nuls2.len(), 4);

    // string equality should pass between the strings
    assert_eq!(all_nuls1, all_nuls2);
    assert_eq!(all_nuls2, all_nuls3);
    assert_eq!(all_nuls3, all_nuls4);

    // all extracted characters in all_nuls are equivalent to each other
    for c1 in all_nuls1.chars()
    {
        for c2 in all_nuls1.chars()
        {
            assert_eq!(c1,c2);
        }
    }

    // testing equality between explicit character literals
    assert_eq!('\0', '\x00');
    assert_eq!('\u{0}', '\x00');
    assert_eq!('\u{0}', '\u{0}');

    // NUL characters should make a difference
    assert!("Hello World" != "Hello \0World");
    assert!("Hello World" != "Hello World\0");
}
