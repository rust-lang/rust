mod parser;
mod byte;
mod byte_string;
mod char;
mod string;

pub use self::{
    byte::parse_byte_literal,
    byte_string::parse_byte_string_literal,
    char::parse_char_literal,
    parser::{CharComponent, CharComponentKind, StringComponent, StringComponentKind},
    string::parse_string_literal,
};
