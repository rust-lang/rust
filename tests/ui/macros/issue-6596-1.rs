macro_rules! e {
    ($inp:ident) => (
        $nonexistent
        //~^ ERROR cannot find macro parameter `$nonexistent` in this scope
    );
}

fn main() {
    e!(foo);
}
