//type t = { a: int };
// type t = { a: bool };
type t = bool;

trait it {
    fn f();
}

impl  of it for t {
    fn f() { }
}

fn main() {
  //    let x = ({a: 4i} as it);
  //   let y = @({a: 4i});
  //    let z = @({a: 4i} as it);
  //    let z = @({a: true} as it);
    let z = @(true as it);
    //  x.f();
    // y.f();
    // (*z).f();
    #error["ok so far..."];
    z.f(); //segfault
}
