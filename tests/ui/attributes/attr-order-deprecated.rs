#[deprecated = "AAA"]
//~^ NOTE also specified here
#[deprecated = "BBB"]
//~^ ERROR multiple `deprecated` attributes
fn deprecated() { }

fn main() {
    deprecated();
    //~^ WARN use of deprecated function `deprecated`: AAA [deprecated]
    //~| NOTE `#[warn(deprecated)]` on by default
}
