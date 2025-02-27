#![warn(clippy::too_many_lines)]
#![allow(clippy::let_unit_value)]

// This function should be considered one line.
fn many_comments_but_one_line_of_code() {
    /* println!("This is good."); */
    // println!("This is good.");
    /* */ // println!("This is good.");
    /* */ // println!("This is good.");
    /* */ // println!("This is good.");
    /* */ // println!("This is good.");
    /* println!("This is good.");
    println!("This is good.");
    println!("This is good."); */
    println!("This is good.");
}

// This should be considered two and a fail.
fn too_many_lines() {
    //~^ too_many_lines
    println!("This is bad.");
    println!("This is bad.");
}

// This should only fail once (#7517).
async fn async_too_many_lines() {
    //~^ too_many_lines
    println!("This is bad.");
    println!("This is bad.");
}

// This should fail only once, without failing on the closure.
fn closure_too_many_lines() {
    //~^ too_many_lines
    let _ = {
        println!("This is bad.");
        println!("This is bad.");
    };
}

// This should be considered one line.
#[rustfmt::skip]
fn comment_starts_after_code() {
    let _ = 5; /* closing comment. */ /*
    this line shouldn't be counted theoretically.
    */
}

// This should be considered one line.
fn comment_after_code() {
    let _ = 5; /* this line should get counted once. */
}

// This should fail since it is technically two lines.
#[rustfmt::skip]
fn comment_before_code() {
//~^ too_many_lines
    let _ = "test";
    /* This comment extends to the front of
    the code but this line should still count. */ let _ = 5;
}

// This should be considered one line.
fn main() {}
