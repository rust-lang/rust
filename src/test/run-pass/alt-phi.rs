

tag thing { a; b; c; }

iter foo() -> int { put 10; }

fn main() {
    let x = true;
    alt a {
      a. { x = true; for each i: int  in foo() { } }
      b. { x = false; }
      c. { x = false; }
    }
}