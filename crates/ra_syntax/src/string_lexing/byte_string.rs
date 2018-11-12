use super::parser::Parser;
use super::StringComponent;

pub fn parse_byte_string_literal(src: &str) -> ByteStringComponentIterator {
    ByteStringComponentIterator {
        parser: Parser::new(src),
        has_closing_quote: false,
    }
}

pub struct ByteStringComponentIterator<'a> {
    parser: Parser<'a>,
    pub has_closing_quote: bool,
}

impl<'a> Iterator for ByteStringComponentIterator<'a> {
    type Item = StringComponent;
    fn next(&mut self) -> Option<StringComponent> {
        if self.parser.pos == 0 {
            assert!(
                self.parser.advance() == 'b',
                "byte string literal should start with a `b`"
            );

            assert!(
                self.parser.advance() == '"',
                "byte string literal should start with a `b`, followed by double quotes"
            );
        }

        if let Some(component) = self.parser.parse_string_component() {
            return Some(component);
        }

        // We get here when there are no char components left to parse
        if self.parser.peek() == Some('"') {
            self.parser.advance();
            self.has_closing_quote = true;
        }

        assert!(
            self.parser.peek() == None,
            "byte string literal should leave no unparsed input: src = {}, pos = {}, length = {}",
            self.parser.src,
            self.parser.pos,
            self.parser.src.len()
        );

        None
    }
}
