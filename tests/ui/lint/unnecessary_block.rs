#![deny(unnecessary_block)]

fn main() {
    {}
    //~^ ERROR unnecessary block [unnecessary_block]
    {}
    //~^ ERROR unnecessary block [unnecessary_block]
}
