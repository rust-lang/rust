export rsrc;

fn foo(_x: i32) {
}

struct rsrc {
  let x: i32;
  drop { foo(self.x); }
}

fn rsrc(x: i32) -> rsrc {
    rsrc {
        x: x
    }
}