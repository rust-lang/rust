//@ edition:2021

async fn foo() {}

fn main() {
    std::mem::size_of_val(foo());
    //~^ ERROR: mismatched types
}
