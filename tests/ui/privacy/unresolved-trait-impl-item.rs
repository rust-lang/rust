//@ edition:2018

trait MyTrait {
    async fn resolved(&self);
    const RESOLVED_WRONG: u8 = 0;
}

impl MyTrait for i32 {
    async fn resolved(&self) {}

    async fn unresolved(&self) {} //~ ERROR method `unresolved` is not a member of trait `MyTrait`
    async fn RESOLVED_WRONG() {} //~ ERROR doesn't match its trait `MyTrait`
}

fn main() {}
