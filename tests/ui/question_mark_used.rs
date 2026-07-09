// non rustfixable
#![warn(clippy::question_mark_used)]

fn other_function() -> Option<i32> {
    Some(32)
}

fn my_function() -> Option<i32> {
    other_function()?;
    //~^ question_mark_used

    None
}

fn main() {}
