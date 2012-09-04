export rsrc;

fn foo(_x: i32) {
}

struct rsrc {
  let x: i32;
  new(x: i32) { self.x = x; }
  drop { foo(self.x); }
}