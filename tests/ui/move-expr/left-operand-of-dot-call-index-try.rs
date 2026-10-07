// Ensure that move exprs can be the left operand of dot, call, index & try ops. We once used to
// incorrectly parse move exprs as prefix exprs instead of bottom ones which used to prevent that.

//@ check-pass
//@ edition: 2018..
#![feature(move_expr)]

fn main() {
    let _ = async || { move(async {}).await }; // Await(Move(_))

    let _ = || -> Result<(), ()> {
        move(((),)).0; // Field(Move(_), ..)
        move(Err(()))?; // Try(Move(_))
        move(drop)(()); // Call(Move(_), ..)
        move([0])[0]; // Index(Move(_), ..)

        // Previously, we would parse this as Field(Return(Move(_)), ..) (`(return move(…)).0`)
        // instead of Return(Field(Move(_), ..)) and thus fail with "no field `0` on type `!`".
        return move((Ok(()),)).0; // OK!
    };
}
