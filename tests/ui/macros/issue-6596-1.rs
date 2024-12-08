macro_rules! e {
    ($inp:ident) => (
        $nonexistent
        //~^ ERROR expected expression, found `$`
    );
}

fn main() {
    e!(foo);
}
