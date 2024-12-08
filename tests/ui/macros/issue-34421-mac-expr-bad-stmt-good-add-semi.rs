macro_rules! make_item {
    ($a:ident) => {
        struct $a;
    }; //~^ ERROR expected expression
       //~| ERROR expected expression
}

fn a() {
    make_item!(A)
}
fn b() {
    make_item!(B)
}

fn main() {}
