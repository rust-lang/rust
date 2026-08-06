//! The closure body must constrain its input before call arguments are coerced.
//! Otherwise the first call fixes the input to `&Box<Pattern>` and rejects the second call.

//@ check-pass

struct Pattern;

fn visit_pattern(_: &Pattern) {}

fn main() {
    let visit_subpattern = |pattern| visit_pattern(pattern);
    let boxed_pattern = Box::new(Pattern);

    // The body constrains `pattern` to `&Pattern`, so this applies a deref coercion.
    visit_subpattern(&boxed_pattern);
    visit_subpattern(&Pattern);
}
