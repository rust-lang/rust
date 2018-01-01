use {Token, TextUnit, SyntaxKind};

use syntax_kinds::*;


pub(crate) enum Event {
    Start { kind: SyntaxKind },
    Finish,
    Token {
        kind: SyntaxKind,
        n_raw_tokens: u8,
    }
}

pub(crate) fn parse<'t>(text: &'t str, raw_tokens: &'t [Token]) -> Vec<Event> {
    let mut parser = Parser::new(text, raw_tokens);
    parse_file(&mut parser);
    parser.events
}

struct Parser<'t> {
    text: &'t str,
    raw_tokens: &'t [Token],
    non_ws_tokens: Vec<(usize, TextUnit)>,

    pos: usize,
    events: Vec<Event>,
}

impl<'t> Parser<'t> {
    fn new(text: &'t str, raw_tokens: &'t [Token]) -> Parser<'t> {
        let mut non_ws_tokens = Vec::new();
        let mut len = TextUnit::new(0);
        for (idx, &token) in raw_tokens.iter().enumerate() {
            match token.kind {
                WHITESPACE | COMMENT => (),
                _ => non_ws_tokens.push((idx, len)),
            }
            len += token.len;
        }

        Parser {
            text,
            raw_tokens,
            non_ws_tokens,

            pos: 0,
            events: Vec::new(),
        }
    }

    fn start(&mut self, kind: SyntaxKind) {
        self.event(Event::Start { kind });
    }
    fn finish(&mut self) {
        self.event(Event::Finish);
    }
    fn event(&mut self, event: Event) {
        self.events.push(event)
    }
}

fn parse_file(p: &mut Parser) {
    p.start(FILE);
    p.finish();
}