//@ proc-macro: group-span.rs
//@ ignore-backends: gcc

fn main() {
    // This code uses `set_span_open`/`set_span_close` for the group,
    // causing both delimiters to have exactly the span of the input delimiter,
    // making for a good error span.
    group_span::different_span! {
        {
            // hello i am a line in between please make sure the span points at the end brace only
        } //~ ERROR expected pattern, found `}`
    };

    // This code uses `set_span` for the whole group, causing both delimiters to have the span
    // of the full block, making for a bad error span.
    group_span::same_span! {
        { //~ ERROR expected pattern, found `}`
            // hello i am a line in between please make sure the span points at the entire block
        }
    };
}
