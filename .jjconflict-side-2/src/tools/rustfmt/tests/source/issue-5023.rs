// rustfmt-wrap_comments: true

// below we try and force a split at a byte in the middle of a multi-byte
// character. The two parts of the first line are constructed such that:
// 1) the entire line is longer than `$comment_width`
// 2) the length of the second part is such that:
//   `$comment_width - 3 + $length` is positive _and_ less than the number of
//   bytes in the multi-byte char

// xxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// 是

/// A comment to test special unicode characters on boundaries
/// 是，是，是，是，是，是，是，是，是，是，是，是  it should break right here this goes to the next line
fn main() {
    if xxx {
        let xxx = xxx
            .into_iter()
            .filter(|(xxx, xxx)| {
                if let Some(x) = Some(1) {
                    // xxxxxxxxxxxxxxxxxx, xxxxxxxxxxxx, xxxxxxxxxxxxxxxxxxxx xxx xxxxxxx, xxxxx xxx
                    // xxxxxxxxxx. xxxxxxxxxxxxxxxx，xxxxxxxxxxxxxxxxx xxx xxxxxxx
                    // 是sdfadsdfxxxxxxxxx，sdfaxxxxxx_xxxxx_masdfaonxxx，
                    if false {
                        return true;
                    }
                }
                false
            })
            .collect();
    }
}
