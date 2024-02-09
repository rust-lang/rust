// edition:2018

struct F;

impl async Fn<()> for F {}
//~^ ERROR expected type, found keyword `async`

fn main() {}
