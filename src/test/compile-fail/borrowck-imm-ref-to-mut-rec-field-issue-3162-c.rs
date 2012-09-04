fn main() {
    let mut _a = 3;
    let _b = &mut _a;
    {
        let _c = &*_b; //~ ERROR illegal borrow unless pure
        _a = 4; //~ NOTE impure due to assigning to mutable local variable
    }
}
