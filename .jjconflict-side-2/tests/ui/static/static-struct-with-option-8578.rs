// https://github.com/rust-lang/rust/issues/8578
//@ check-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

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
