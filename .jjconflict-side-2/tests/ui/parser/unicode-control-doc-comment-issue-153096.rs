//@ edition: 2024

#[allow(unused)]
/// ⁨א⁩, ⁨ב⁩, ⁨ג⁩, ⁨ד⁩, ⁨ה⁩
//~^ ERROR unicode codepoint changing visible direction of text present in doc comment
fn foo() {}

fn main() {}
