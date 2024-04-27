// issue: 112732

// `}` is typoed since it is interpreted as a fill character rather than a closing bracket

fn main() {
    println!("Hello, world! {0:}<3", 2);
    //~^ ERROR invalid format string: expected `'}'` but string was terminated
}
