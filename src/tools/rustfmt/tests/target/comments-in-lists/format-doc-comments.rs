// rustfmt-format_code_in_doc_comments: true

// https://github.com/rust-lang/rustfmt/issues/4420
enum Minimal {
    Example,
    //[thisisremoved thatsleft
    // canbeanything
}

struct Minimal2 {
    Example: usize,
    //[thisisremoved thatsleft
    // canbeanything
}

pub enum E {
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
    Variant1,
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
    Variant2,
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
}

pub enum E2 {
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
}

pub struct S {
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
    some_field: usize,
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
    last_field: usize,
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
}

pub struct S2 {
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
}

fn foo(
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
    a: usize,
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
    b: usize,
    // Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
) -> usize {
    5
}

fn foo2(// Expand as needed, numbers should be ascending according to the stage
    // through the inclusion pipeline, or according to the descriptions
) -> usize {
    5
}

fn main() {
    let v = vec![
        // Expand as needed, numbers should be ascending according to the stage
        // through the inclusion pipeline, or according to the descriptions
        1,
        // Expand as needed, numbers should be ascending according to the stage
        // through the inclusion pipeline, or according to the descriptions
        2,
        // Expand as needed, numbers should be ascending according to the stage
        // through the inclusion pipeline, or according to the descriptions
    ];

    let v2: Vec<i32> = vec![
        // Expand as needed, numbers should be ascending according to the stage
        // through the inclusion pipeline, or according to the descriptions
    ];

    match a {
        // Expand as needed, numbers should be ascending according to the stage
        // through the inclusion pipeline, or according to the descriptions
        b => c,
        // Expand as needed, numbers should be ascending according to the stage
        // through the inclusion pipeline, or according to the descriptions
        d => e,
        // Expand as needed, numbers should be ascending according to the stage
        // through the inclusion pipeline, or according to the descriptions
    }
}
