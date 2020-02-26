//! Defines token tags we use for syntax highlighting.
//! A tag is not unlike a CSS class.

use std::fmt;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct HighlightTag(&'static str);

impl fmt::Display for HighlightTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.0, f)
    }
}

#[rustfmt::skip]
impl HighlightTag {
    pub const FIELD: HighlightTag              = HighlightTag("field");
    pub const FUNCTION: HighlightTag           = HighlightTag("function");
    pub const MODULE: HighlightTag             = HighlightTag("module");
    pub const CONSTANT: HighlightTag           = HighlightTag("constant");
    pub const MACRO: HighlightTag              = HighlightTag("macro");

    pub const VARIABLE: HighlightTag           = HighlightTag("variable");
    pub const VARIABLE_MUT: HighlightTag       = HighlightTag("variable.mut");

    pub const TYPE: HighlightTag               = HighlightTag("type");
    pub const TYPE_BUILTIN: HighlightTag       = HighlightTag("type.builtin");
    pub const TYPE_SELF: HighlightTag          = HighlightTag("type.self");
    pub const TYPE_PARAM: HighlightTag         = HighlightTag("type.param");
    pub const TYPE_LIFETIME: HighlightTag      = HighlightTag("type.lifetime");

    pub const LITERAL_BYTE: HighlightTag       = HighlightTag("literal.byte");
    pub const LITERAL_NUMERIC: HighlightTag    = HighlightTag("literal.numeric");
    pub const LITERAL_CHAR: HighlightTag       = HighlightTag("literal.char");

    pub const LITERAL_COMMENT: HighlightTag    = HighlightTag("comment");
    pub const LITERAL_STRING: HighlightTag     = HighlightTag("string");
    pub const LITERAL_ATTRIBUTE: HighlightTag  = HighlightTag("attribute");

    pub const KEYWORD: HighlightTag            = HighlightTag("keyword");
    pub const KEYWORD_UNSAFE: HighlightTag     = HighlightTag("keyword.unsafe");
    pub const KEYWORD_CONTROL: HighlightTag    = HighlightTag("keyword.control");
}
