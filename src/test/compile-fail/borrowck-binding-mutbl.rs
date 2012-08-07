fn impure(_v: &[int]) {
}

fn main() {
    let x = {mut f: ~[3]};

    match x {
      {f: ref mut v} => {
        impure(*v); //~ ERROR illegal borrow unless pure
        //~^ NOTE impure due to access to impure function
      }
    }
}
