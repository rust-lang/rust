pure fn pure_borrow(_x: &int, _y: ()) {}

fn test1(x: @mut ~int) {
    // Here, evaluating the second argument actually invalidates the
    // first borrow, even though it occurs outside of the scope of the
    // borrow!
    pure_borrow(*x, *x = ~5);  //~ ERROR illegal borrow unless pure: unique value in aliasable, mutable location
    //~^ NOTE impure due to assigning to dereference of mutable @ pointer
}

fn test2() {
    let mut x = ~1;

    // Same, but for loanable data:

    pure_borrow(x, x = ~5);  //~ ERROR assigning to mutable local variable prohibited due to outstanding loan
    //~^ NOTE loan of mutable local variable granted here

    copy x;
}

fn main() {
}