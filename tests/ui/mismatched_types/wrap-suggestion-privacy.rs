mod inner {
    pub struct Wrapper<T>(T);
}

fn needs_wrapper(t: inner::Wrapper<i32>) {}
fn needs_wrapping(t: std::num::Wrapping<i32>) {}
fn needs_ready(t: std::future::Ready<i32>) {}

fn main() {
    // Suggest wrapping expression because type is local
    // and its privacy can be easily changed
    needs_wrapper(0);
    //~^ ERROR mismatched types
    //~| HELP  try wrapping the expression in `inner::Wrapper`

    // Suggest wrapping expression because field is accessible
    needs_wrapping(0);
    //~^ ERROR mismatched types
    //~| HELP  try wrapping the expression in `std::num::Wrapping`

    // Do not suggest wrapping expression
    needs_ready(Some(0));
    //~^ ERROR mismatched types
}
