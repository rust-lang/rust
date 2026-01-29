fn main() {
    const EMPTY_ARRAY: _ = [];
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants [E0121]
    //~| ERROR type annotations needed
}
