use super::Event;
use super::parser::Parser;

use syntax_kinds::*;

pub fn parse_file(p: &mut Parser) {
    p.start(FILE);
    p.finish();
}