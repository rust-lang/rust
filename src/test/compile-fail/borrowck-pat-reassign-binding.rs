// xfail-fast  (compile-flags unsupported on windows)
// compile-flags:--borrowck=err
// xfail-pretty -- comments are infaithfully preserved

fn main() {
    let mut x: option<int> = none;
    alt x { //! NOTE loan of mutable local variable granted here
      none {}
      some(i) {
        // Not ok: i is an outstanding ptr into x.
        x = some(i+1); //! ERROR assigning to mutable local variable prohibited due to outstanding loan
      }
    }
    copy x; // just to prevent liveness warnings
}
