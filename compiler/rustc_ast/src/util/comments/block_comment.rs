/*!
 * Block comment helpers.
 */

#[cfg(test)]
mod tests;

/********************************************************
 * Skip lines based on the following rules:
 *
 * * Skip first line of all stars ("*").
 * * Skip consecutive empty lines top-bottom.
 * * Skip consecutive empty lines bottom-top.
 * * Skip last line contains pattern "^ ?\**$" in regex.
 *******************************************************/
pub fn vertical_trim<'arr, 'row: 'arr>(lines: &'arr [&'row str]) -> &'arr [&'row str] {
    let mut region = lines;
    if let [first, tail @ ..] = region {
        // Skip first line of all-stars.
        if first.bytes().all(|c| c == b'*') {
            region = tail;
        }
    }

    // Skip consecutive empty lines.
    loop {
        match region {
            [first, tail @ ..] if first.trim().is_empty() => region = tail,
            _ => break,
        }
    }

    // Skip last line contains pattern "^ ?*\**" in regex.
    if let [head @ .., last] = region {
        let s = match last.as_bytes() {
            [b' ', tail @ ..] => tail,
            all => all,
        };
        if s.iter().all(|&c| c == b'*') {
            region = head;
        }
    }

    // Skip consecutive empty lines from last line backward.
    loop {
        match region {
            [head @ .., last] if last.trim().is_empty() => region = head,
            _ => break,
        }
    }

    region
}

/// Trim all "\s*\*" prefix from comment: all or nothing.
///
/// For example,
/// ```text
///   * one two three four five ...
///   * one two three four five ...
///   * one two three four five ...
/// ```
/// will be trimmed to
/// ```text
///  one two three four five ...
///  one two three four five ...
///  one two three four five ...
/// ```
pub fn horizontal_trim<'arr, 'row: 'arr>(lines: &'arr [&'row str]) -> Option<Vec<&'row str>> {
    let prefix = match lines {
        [first, ..] => get_prefix(first)?,
        _ => return None,
    };

    if lines.iter().any(|l| !l.starts_with(prefix)) {
        return None;
    }

    let lines = lines
        .iter()
        // SAFETY: All lines have been checked if it starts with prefix
        .map(|l| unsafe { l.get_unchecked(prefix.len()..) })
        .collect();
    Some(lines)
}

/// Get the prefix with pattern "\s*\*" of input `s`.
fn get_prefix(s: &str) -> Option<&str> {
    let mut bytes = s.as_bytes();
    let dst: *const u8 = loop {
        match bytes {
            [b' ' | b'\t', end @ ..] => bytes = end,
            [b'*', end @ ..] => break end.as_ptr(),
            _ => return None,
        }
    };
    let prefix = unsafe {
        // SAFETY: Two invariants are followed.
        // * length of `prefix` is the diff of two pointer from the same str `s`.
        // * lifetime of `prefix` is the same as argument `s`.
        let src: *const u8 = s.as_ptr();
        let len = dst as usize - src as usize;
        let slice = std::slice::from_raw_parts(src, len);
        std::str::from_utf8_unchecked(slice)
    };
    Some(prefix)
}
