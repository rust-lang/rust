// check-pass

#![feature(lint_reasons)]

#![expect(unconditional_panic, unused, reason = "Don't trigger because `unused` was triggered")]
fn main() {
    let x = 0;
}
