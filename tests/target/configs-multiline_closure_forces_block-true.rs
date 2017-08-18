// rustfmt-multiline_closure_forces_block: true
// Option forces multiline closure bodies to be wrapped in a block

fn main() {
    result.and_then(|maybe_value| {
        match maybe_value {
            None => Err("oops"),
            Some(value) => Ok(1),
        }
    });
}
