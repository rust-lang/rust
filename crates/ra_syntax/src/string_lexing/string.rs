use super::parser::Parser;
use super::StringComponent;

pub fn parse_string_literal(src: &str) -> StringComponentIterator {
    StringComponentIterator {
        parser: Parser::new(src),
        has_closing_quote: false,
    }
}

pub struct StringComponentIterator<'a> {
    parser: Parser<'a>,
    pub has_closing_quote: bool,
}

impl<'a> Iterator for StringComponentIterator<'a> {
    type Item = StringComponent;
    fn next(&mut self) -> Option<StringComponent> {
        if self.parser.pos == 0 {
            assert!(
                self.parser.advance() == '"',
                "string literal should start with double quotes"
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
            "string literal should leave no unparsed input: src = {}, pos = {}, length = {}",
            self.parser.src,
            self.parser.pos,
            self.parser.src.len()
        );

        None
    }
}
