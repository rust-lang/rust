use {Token, File, FileBuilder, Sink, SyntaxKind};

use syntax_kinds::*;

mod event_parser;
use self::event_parser::Event;


pub fn parse(text: String, tokens: &[Token]) -> File {
    let events = event_parser::parse(&text, tokens);
    from_events_to_file(text, events)
}

fn from_events_to_file(text: String, events: Vec<Event>) -> File {
    let mut builder = FileBuilder::new(text);
    for event in events {
        match event {
            Event::Start { kind } => builder.start_internal(kind),
            Event::Finish => builder.finish_internal(),
            Event::Token { .. } => unimplemented!(),
        }
    }
    builder.finish()
}
