mod bar {
    enum foo {
        alpha,
        beta,
        charlie
    }
}

fn main() {
    import bar::{alpha, charlie};
    match alpha {
      alpha | beta => {} //~ ERROR: inconsistent number of bindings
      charlie => {}
    }
}
