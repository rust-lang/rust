//@edition: 2021

#![allow(clippy::let_and_return)]

// issue #11814
fn block_in_match_expr(num: i32) -> i32 {
    match {
        //~^ ERROR: in a `match` scrutinee, avoid complex blocks or closures with blocks; instead, move the block or closure higher and bind it with a `let`
        let opt = Some(2);
        opt
    } {
        Some(0) => 1,
        Some(n) => num * 2,
        None => 0,
    };

    match unsafe {
        let hearty_hearty_hearty = vec![240, 159, 146, 150];
        String::from_utf8_unchecked(hearty_hearty_hearty).as_str()
    } {
        "💖" => 1,
        "what" => 2,
        _ => 3,
    }
}
