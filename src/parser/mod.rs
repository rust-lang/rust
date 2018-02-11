use {File, SyntaxKind, Token};

use syntax_kinds::*;

#[macro_use]
mod parser;
mod input;
mod event;
mod grammar;
use self::event::Event;

/// Parse a sequence of tokens into the representative node tree
pub fn parse(text: String, tokens: &[Token]) -> File {
    let events = {
        let input = input::ParserInput::new(&text, tokens);
        let mut parser = parser::Parser::new(&input);
        grammar::file(&mut parser);
        parser.into_events()
    };
    event::to_file(text, tokens, events)
}

fn is_insignificant(kind: SyntaxKind) -> bool {
    match kind {
        WHITESPACE | COMMENT => true,
        _ => false,
    }
}

impl<'p> parser::Parser<'p> {
    fn at(&self, kind: SyntaxKind) -> bool {
        self.current() == kind
    }

    fn err_and_bump(&mut self, message: &str) {
        let err = self.start();
        self.error(message);
        self.bump();
        err.complete(self, ERROR);
    }

    fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            self.error(format!("expected {:?}", kind));
            false
        }
    }

    fn eat(&mut self, kind: SyntaxKind) -> bool {
        self.at(kind) && {
            self.bump();
            true
        }
    }
}
