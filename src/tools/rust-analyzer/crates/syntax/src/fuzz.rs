//! Some infrastructure for fuzzy testing.
//!
//! We don't normally run fuzzying, so this is hopelessly bitrotten :(

use std::str::{self, FromStr};

use parser::Edition;

use crate::{AstNode, SourceFile, TextRange, validation};

fn check_file_invariants(file: &SourceFile) {
    let root = file.syntax();
    validation::validate_block_structure(root);
}

pub fn check_parser(text: &str) {
    let file = SourceFile::parse(text, Edition::CURRENT);
    check_file_invariants(&file.tree());
}

#[derive(Debug, Clone)]
pub struct CheckReparse {
    text: String,
    delete: TextRange,
    insert: String,
    edited_text: String,
}

impl CheckReparse {
    pub fn from_data(data: &[u8]) -> Option<Self> {
        const PREFIX: &str = "fn main(){\n\t";
        const SUFFIX: &str = "\n}";

        let data = str::from_utf8(data).ok()?;
        let mut lines = data.lines();
        let delete_start = usize::from_str(lines.next()?).ok()? + PREFIX.len();
        let delete_len = usize::from_str(lines.next()?).ok()?;
        let insert = lines.next()?.to_owned();
        let text = lines.collect::<Vec<_>>().join("\n");
        let text = format!("{PREFIX}{text}{SUFFIX}");
        text.get(delete_start..delete_start.checked_add(delete_len)?)?; // make sure delete is a valid range
        let delete =
            TextRange::at(delete_start.try_into().unwrap(), delete_len.try_into().unwrap());
        let edited_text =
            format!("{}{}{}", &text[..delete_start], &insert, &text[delete_start + delete_len..]);
        Some(CheckReparse { text, insert, delete, edited_text })
    }

    #[allow(clippy::print_stderr)]
    pub fn run(&self) {
        let parse = SourceFile::parse(&self.text, Edition::CURRENT);
        let new_parse = parse.reparse(self.delete, &self.insert, Edition::CURRENT);
        check_file_invariants(&new_parse.tree());
        assert_eq!(&new_parse.tree().syntax().text().to_string(), &self.edited_text);
        let full_reparse = SourceFile::parse(&self.edited_text, Edition::CURRENT);
        for (a, b) in
            new_parse.tree().syntax().descendants().zip(full_reparse.tree().syntax().descendants())
        {
            if (a.kind(), a.text_range()) != (b.kind(), b.text_range()) {
                eprint!("original:\n{:#?}", parse.tree().syntax());
                eprint!("reparsed:\n{:#?}", new_parse.tree().syntax());
                eprint!("full reparse:\n{:#?}", full_reparse.tree().syntax());
                assert_eq!(
                    format!("{a:?}"),
                    format!("{b:?}"),
                    "different syntax tree produced by the full reparse"
                );
            }
        }
        // FIXME
        // assert_eq!(new_file.errors(), full_reparse.errors());
    }
}
