fn issue_17068() {
    fn nop() {}

    match Some(()) {
        Some(()) => match {
            //~^ ERROR: in a `match` scrutinee, avoid complex blocks or closures with blocks; instead, move the block or closure higher and bind it with a `let`
            nop();
            42
        } {
            42 => nop(),
            _ => nop(),
        },
        None => nop(),
    }

    let _x = if {
        //~^ ERROR: in an `if` condition, avoid complex blocks or closures with blocks; instead, move the block or closure higher and bind it with a `let`
        let v = 1;
        v == 1
    } {
        1
    } else {
        2
    };
}
