#![allow(clippy::needless_if, dead_code, unused_must_use, clippy::double_ended_iterator_last)]

fn main() {}

#[allow(clippy::unnecessary_operation)]
fn starts_with() {
    "".chars().next() == Some(' ');
    //~^ chars_next_cmp
    Some(' ') != "".chars().next();
    //~^ chars_next_cmp

    // Ensure that suggestion is escaped correctly
    "".chars().next() == Some('\n');
    //~^ chars_next_cmp
    Some('\n') != "".chars().next();
    //~^ chars_next_cmp
}

fn chars_cmp_with_unwrap() {
    let s = String::from("foo");
    if s.chars().next().unwrap() == 'f' {
        //~^ chars_next_cmp
        // s.starts_with('f')
        // Nothing here
    }
    if s.chars().next_back().unwrap() == 'o' {
        //~^ chars_last_cmp
        // s.ends_with('o')
        // Nothing here
    }
    if s.chars().last().unwrap() == 'o' {
        //~^ chars_last_cmp
        // s.ends_with('o')
        // Nothing here
    }
    if s.chars().next().unwrap() != 'f' {
        //~^ chars_next_cmp
        // !s.starts_with('f')
        // Nothing here
    }
    if s.chars().next_back().unwrap() != 'o' {
        //~^ chars_last_cmp
        // !s.ends_with('o')
        // Nothing here
    }
    if s.chars().last().unwrap() != '\n' {
        //~^ chars_last_cmp
        // !s.ends_with('o')
        // Nothing here
    }
}

#[allow(clippy::unnecessary_operation)]
fn ends_with() {
    "".chars().last() == Some(' ');
    //~^ chars_last_cmp
    Some(' ') != "".chars().last();
    //~^ chars_last_cmp
    "".chars().next_back() == Some(' ');
    //~^ chars_last_cmp
    Some(' ') != "".chars().next_back();
    //~^ chars_last_cmp

    // Ensure that suggestion is escaped correctly
    "".chars().last() == Some('\n');
    //~^ chars_last_cmp
    Some('\n') != "".chars().last();
    //~^ chars_last_cmp
}
