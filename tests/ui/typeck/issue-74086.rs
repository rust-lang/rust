fn main() {
    static BUG: fn(_) -> u8 = |_| 8;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions [E0121]
    //~| ERROR the placeholder `_` is not allowed within types on item signatures for static items
}
