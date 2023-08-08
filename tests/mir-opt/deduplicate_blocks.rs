// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DeduplicateBlocks

// EMIT_MIR deduplicate_blocks.is_line_doc_comment_2.DeduplicateBlocks.diff
pub const fn is_line_doc_comment_2(s: &str) -> bool {
    match s.as_bytes() {
        [b'/', b'/', b'/', b'/', ..] => false,
        [b'/', b'/', b'/', ..] => true,
        [b'/', b'/', b'!', ..] => true,
        _ => false,
    }
}

fn main() {
    is_line_doc_comment_2("asd");
}
