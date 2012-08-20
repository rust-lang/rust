// xfail-pretty -- comments are infaithfully preserved

fn main() {
    let mut x = None;
    match x { //~ NOTE loan of mutable local variable granted here
      None => {
        // It is ok to reassign x here, because there is in
        // fact no outstanding loan of x!
        x = Some(0);
      }
      Some(ref _i) => {
        x = Some(1); //~ ERROR assigning to mutable local variable prohibited due to outstanding loan
      }
    }
    copy x; // just to prevent liveness warnings
}
