#![warn(clippy::unnecessary_safety_comment)]

fn foo() -> i32 {
    // SAFETY: fail ONLY if `accept-comment-above-attribute = false`
    #[must_use]
    return 33;
    //~^ unnecessary_safety_comment
}

fn main() {}
