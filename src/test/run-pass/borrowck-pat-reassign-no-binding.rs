// xfail-fast  (compile-flags unsupported on windows)
// compile-flags:--borrowck=err

fn main() {
    let mut x = none;
    alt x {
      none {
        // It is ok to reassign x here, because there is in
        // fact no outstanding loan of x!
        x = some(0);
      }
      some(_) { }
    }
}
