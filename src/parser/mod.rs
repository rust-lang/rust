use {Token, File, FileBuilder, Sink};

use syntax_kinds::*;


pub fn parse(text: String, tokens: &[Token]) -> File {
    let mut builder = FileBuilder::new(text);
    builder.start_internal(FILE);
    builder.finish_internal();
    builder.finish()
}