macro_rules! e {
    ($inp:ident) => (
        $nonexistent
        //~^ ERROR cannot find macro parameter `$nonexistent` in this scope
    );
}

macro_rules! m {
    () => (
        $x
        //~^ ERROR cannot find macro parameter `$x` in this scope
    );
}

fn main() {
    e!(foo);
    m!();
}
