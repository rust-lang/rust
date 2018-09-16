use {TextRange, TextUnit};

pub fn contains_offset_nonstrict(range: TextRange, offset: TextUnit) -> bool {
    range.start() <= offset && offset <= range.end()
}

pub fn is_subrange(range: TextRange, subrange: TextRange) -> bool {
    range.start() <= subrange.start() && subrange.end() <= range.end()
}

pub fn intersect(r1: TextRange, r2: TextRange) -> Option<TextRange> {
    let start = r1.start().max(r2.start());
    let end = r1.end().min(r2.end());
    if start <= end {
        Some(TextRange::from_to(start, end))
    } else {
        None
    }
}

pub fn replace_range(mut text: String, range: TextRange, replace_with: &str) -> String {
    let start = u32::from(range.start()) as usize;
    let end = u32::from(range.end()) as usize;
    text.replace_range(start..end, replace_with);
    text
}