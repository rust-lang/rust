#[allow(non_camel_case_types)]

mod bar {
    pub enum foo {
        alpha,
        beta,
        charlie
    }
}

fn main() {
    use bar::foo::{alpha, charlie};
    match alpha {
      alpha | beta => {} //~ ERROR variable `beta` is not bound in all patterns
      //~^ ERROR `beta` is named the same as one of the variants
      charlie => {}
    }
}
