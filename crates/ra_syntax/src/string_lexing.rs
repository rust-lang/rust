mod parser;
mod string;

pub use self::{
    parser::{StringComponent, StringComponentKind},
    string::{parse_string_literal, parse_char_literal, parse_byte_literal, parse_byte_string_literal},
};
