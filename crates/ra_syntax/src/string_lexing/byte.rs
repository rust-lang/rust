use super::parser::Parser;
use super::CharComponent;

pub fn parse_byte_literal(src: &str) -> ByteComponentIterator {
    ByteComponentIterator {
        parser: Parser::new(src),
        has_closing_quote: false,
    }
}

pub struct ByteComponentIterator<'a> {
    parser: Parser<'a>,
    pub has_closing_quote: bool,
}

impl<'a> Iterator for ByteComponentIterator<'a> {
    type Item = CharComponent;
    fn next(&mut self) -> Option<CharComponent> {
        if self.parser.pos == 0 {
            assert!(
                self.parser.advance() == 'b',
                "Byte literal should start with a `b`"
            );

            assert!(
                self.parser.advance() == '\'',
                "Byte literal should start with a `b`, followed by a quote"
            );
        }

        if let Some(component) = self.parser.parse_char_component() {
            return Some(component);
        }

        // We get here when there are no char components left to parse
        if self.parser.peek() == Some('\'') {
            self.parser.advance();
            self.has_closing_quote = true;
        }

        assert!(
            self.parser.peek() == None,
            "byte literal should leave no unparsed input: src = {}, pos = {}, length = {}",
            self.parser.src,
            self.parser.pos,
            self.parser.src.len()
        );

        None
    }
}
