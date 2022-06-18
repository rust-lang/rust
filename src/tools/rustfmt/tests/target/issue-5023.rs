// rustfmt-wrap_comments: true

/// A comment to test special unicode characters on boundaries
/// 是，是，是，是，是，是，是，是，是，是，是，是  it should break right here
/// this goes to the next line
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
