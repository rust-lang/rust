// rustfmt-indent_style: Visual
fn main() {
    match option {
        None => some_function(first_reasonably_long_argument,
                              second_reasonably_long_argument),
    }
}

fn main() {
    match option {
        None => {
            some_function(first_reasonably_long_argument,
                          second_reasonably_long_argument)
        }
    }
}
