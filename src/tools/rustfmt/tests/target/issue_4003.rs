//@ rustfmt-match_arm_blocks: false

fn main() {
    match pos_iter.peek() {
        Some(pos) if pos.0 != 0 =>
            add_paragraphs_to_map(&doc[0..pos.0], 0, current_level, &mut paragraphs),
        None => add_paragraphs_to_map(doc, 0, current_level, &mut paragraphs),
        _ => (),
    }
}
