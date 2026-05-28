#[link(name = "foo")]
#[link(
//~^ ERROR malformed `link` attribute input
    name = "bar",
    kind = "static",
    modifiers = "+whole-archive,-whole-archive",
    //~^ ERROR multiple `whole-archive` modifiers in a single `modifiers` argument
    modifiers = "+bundle"
)]
extern "C" {}

fn main() {}
