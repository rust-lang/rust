// run-pass
// compile-flags:--test -O

#[test]
#[should_panic(expected = "creating inhabited type")]
fn test() {
    FontLanguageOverride::system_font(SystemFont::new());
}

pub enum FontLanguageOverride {
    Normal,
    Override(&'static str),
    System(SystemFont)
}

pub enum SystemFont {}

impl FontLanguageOverride {
    fn system_font(f: SystemFont) -> Self {
        FontLanguageOverride::System(f)
    }
}

impl SystemFont {
    fn new() -> Self {
        panic!("creating inhabited type")
    }
}
