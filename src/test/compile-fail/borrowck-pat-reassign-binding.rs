// xfail-pretty -- comments are infaithfully preserved

fn main() {
    let mut x: option<int> = none;
    match x { //~ NOTE loan of mutable local variable granted here
      none => {}
      some(i) => {
        // Not ok: i is an outstanding ptr into x.
        x = some(i+1); //~ ERROR assigning to mutable local variable prohibited due to outstanding loan
      }
    }
    copy x; // just to prevent liveness warnings
}
