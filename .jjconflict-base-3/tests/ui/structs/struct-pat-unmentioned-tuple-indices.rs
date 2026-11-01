// On unmentioned tuple indices in struct patterns, don't suggest turning the pattern into a
// tuple struct pattern and keep the struct pattern in the suggestion.
// issue: <https://github.com/rust-lang/rust/issues/108284>

struct S(i32, f32);
enum E { V(i32, f32) }

fn main() {
    let S { 0: _ } = S(1, 2.2); //~ ERROR: pattern does not mention field `1`
    let E::V { 0: _ } = E::V(1, 2.2); //~ ERROR: pattern does not mention field `1`
}
