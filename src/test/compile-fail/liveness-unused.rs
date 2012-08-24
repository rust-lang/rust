fn f1(x: int) {
    //~^ WARNING unused variable: `x`
}

fn f1b(&x: int) {
    //~^ WARNING unused variable: `x`
}

fn f2() {
    let x = 3;
    //~^ WARNING unused variable: `x`
}

fn f3() {
    let mut x = 3;
    //~^ WARNING variable `x` is assigned to, but never used
    x += 4;
    //~^ WARNING value assigned to `x` is never read
}

fn f3b() {
    let mut z = 3;
    //~^ WARNING variable `z` is assigned to, but never used
    loop {
        z += 4;
    }
}

fn f4() {
    match some(3) {
      some(i) => {
        //~^ WARNING unused variable: `i`
      }
      none => {}
    }
}

enum tri {
    a(int), b(int), c(int)
}

fn f4b() -> int {
    match a(3) {
      a(i) | b(i) | c(i) => {
        i
      }
    }
}

// leave this in here just to trigger compile-fail:
struct r {
    let x: ();
    new() { self.x = (); }
    drop {}
}
fn main() {
    let x = r();
    fn@() { copy x; }; //~ ERROR copying a noncopyable value
}
