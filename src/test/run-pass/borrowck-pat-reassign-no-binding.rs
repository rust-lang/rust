fn main() {
    let mut x = none;
    match x {
      none => {
        // It is ok to reassign x here, because there is in
        // fact no outstanding loan of x!
        x = some(0);
      }
      some(_) => { }
    }
}
