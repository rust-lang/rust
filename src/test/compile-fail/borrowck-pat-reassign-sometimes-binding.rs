// compile-flags:--borrowck=err
// xfail-pretty -- comments are infaithfully preserved

fn main() {
    let mut x = none;
    alt x { //! NOTE loan of mutable local variable granted here
      none {
        // It is ok to reassign x here, because there is in
        // fact no outstanding loan of x!
        x = some(0);
      }
      some(i) {
        x = some(1); //! ERROR cannot assign to mutable local variable due to outstanding loan
      }
    }
}
