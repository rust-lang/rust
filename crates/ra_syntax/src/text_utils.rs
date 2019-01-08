use crate::TextRange;

pub fn replace_range(mut text: String, range: TextRange, replace_with: &str) -> String {
    let start = u32::from(range.start()) as usize;
    let end = u32::from(range.end()) as usize;
    text.replace_range(start..end, replace_with);
    text
}
