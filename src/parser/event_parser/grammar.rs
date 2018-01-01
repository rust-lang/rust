use super::Event;
use super::parser::Parser;

use syntax_kinds::*;

pub fn file(p: &mut Parser) {
    p.start(FILE);
    //TODO: parse_shebang
    p.finish();
}