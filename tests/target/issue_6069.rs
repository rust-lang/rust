// `Foó` as written here ends with ASCII 6F `'o'` followed by `'\u{0301}'` COMBINING ACUTE ACCENT.
// The compiler normalizes that combination to NFC form, `'\u{00F3}'` LATIN SMALL LETTER O WITH ACUTE.
trait Foó: Bar {}
