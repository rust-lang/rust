extern "C" {
    fn foo(...);
//~^ ERROR C-variadic function must be declared with at least one named argument
}

fn main() {}
