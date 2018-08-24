// pretty-expanded FIXME #23616

pub struct UninterpretedOption_NamePart {
    name_part: Option<String>,
}

impl<'a> UninterpretedOption_NamePart {
    pub fn default_instance() -> &'static UninterpretedOption_NamePart {
        static instance: UninterpretedOption_NamePart = UninterpretedOption_NamePart {
            name_part: None,
        };
        &instance
    }
}

pub fn main() {}
