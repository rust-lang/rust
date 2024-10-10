mod m {
    pub let answer = 42;
    //~^ ERROR visibility `pub` is not followed by an item
    //~| ERROR expected item, found keyword `let`
}
