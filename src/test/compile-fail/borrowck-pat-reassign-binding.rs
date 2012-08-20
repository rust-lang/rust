// xfail-pretty -- comments are infaithfully preserved

fn main() {
    let mut x: Option<int> = None;
    match x { //~ NOTE loan of mutable local variable granted here
      None => {}
      Some(ref i) => {
        // Not ok: i is an outstanding ptr into x.
        x = Some(*i+1); //~ ERROR assigning to mutable local variable prohibited due to outstanding loan
      }
    }
    copy x; // just to prevent liveness warnings
}
