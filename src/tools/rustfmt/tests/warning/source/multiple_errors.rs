#[rustfmt_skip]
fn deprecated_skip() {}

#[rustfmt::invalid]
fn invalid_attribute() {}

fn this_function_name_is_intentionally_long_enough_to_exceed_the_default_one_hundred_character_maximum_width() {}

fn lost_comment() { let _ = 1 /* This comment cannot be retained by the expression formatter. */ + 2; }

/// This doc comment has trailing whitespace.   
fn trailing_whitespace() {}
