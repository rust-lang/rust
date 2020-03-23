use {
    std::{borrow::Cow, ops::Deref},
    text_size::*,
};

struct StringLike<'a>(&'a str);

impl Deref for StringLike<'_> {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[test]
fn main() {
    let s = "";
    let _ = TextSize::of(&s);

    let s = String::new();
    let _ = TextSize::of(&s);

    let s = Cow::Borrowed("");
    let _ = TextSize::of(&s);

    let s = Cow::Owned(String::new());
    let _ = TextSize::of(&s);

    let s = StringLike("");
    let _ = TextSize::of(&s);
}
