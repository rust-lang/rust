fn main() {
    static BUG: fn(_) -> u8 = |_| 8;
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures [E0121]
}
