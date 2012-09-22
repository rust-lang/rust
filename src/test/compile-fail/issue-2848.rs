mod bar {
    #[legacy_exports];
    enum foo {
        alpha,
        beta,
        charlie
    }
}

fn main() {
    use bar::{alpha, charlie};
    match alpha {
      alpha | beta => {} //~ ERROR variable `beta` from pattern #2 is not bound in pattern #1
      charlie => {}
    }
}
