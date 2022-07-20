//! Generate minimal `TextEdit`s from different text versions
use dissimilar::Chunk;
use ide::{TextEdit, TextRange, TextSize};

pub(crate) fn diff(left: &str, right: &str) -> TextEdit {
    let chunks = dissimilar::diff(left, right);
    textedit_from_chunks(chunks)
}

fn textedit_from_chunks(chunks: Vec<dissimilar::Chunk<'_>>) -> TextEdit {
    let mut builder = TextEdit::builder();
    let mut pos = TextSize::default();

    let mut chunks = chunks.into_iter().peekable();
    while let Some(chunk) = chunks.next() {
        if let (Chunk::Delete(deleted), Some(&Chunk::Insert(inserted))) = (chunk, chunks.peek()) {
            chunks.next().unwrap();
            let deleted_len = TextSize::of(deleted);
            builder.replace(TextRange::at(pos, deleted_len), inserted.into());
            pos += deleted_len;
            continue;
        }

        match chunk {
            Chunk::Equal(text) => {
                pos += TextSize::of(text);
            }
            Chunk::Delete(deleted) => {
                let deleted_len = TextSize::of(deleted);
                builder.delete(TextRange::at(pos, deleted_len));
                pos += deleted_len;
            }
            Chunk::Insert(inserted) => {
                builder.insert(pos, inserted.into());
            }
        }
    }
    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_applies() {
        let mut original = String::from("fn foo(a:u32){\n}");
        let result = "fn foo(a: u32) {}";
        let edit = diff(&original, result);
        edit.apply(&mut original);
        assert_eq!(original, result);
    }
}
