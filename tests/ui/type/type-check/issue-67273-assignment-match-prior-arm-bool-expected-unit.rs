fn main() {
    let mut i: i64;
    // Expected type is an inference variable `?T`
    // because the `match` is used as a statement.
    // This is the "initial" type of the `coercion`.
    match i {
        // Add `bool` to the overall `coercion`.
        0 => true,

        // Necessary to cause the ICE:
        1 => true,

        // Suppose that we had `let _: bool = match i { ... }`.
        // In that case, as the expected type would be `bool`,
        // we would suggest `i == 1` as a fix.
        //
        // However, no type error happens when checking `i = 1` because `expected == ?T`,
        // which will unify with `typeof(i = 1) == ()`.
        //
        // However, in #67273, we would delay the unification of this arm with the above
        // because we used the hitherto accumulated coercion as opposed to the "initial" type.
        2 => i = 1,
        //~^ ERROR `match` arms have incompatible types

        _ => (),
    }
}
