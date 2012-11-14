export rsrc;

fn foo(_x: i32) {
}

struct rsrc {
  x: i32,
}

impl rsrc : Drop {
    fn finalize() {
        foo(self.x);
    }
}

fn rsrc(x: i32) -> rsrc {
    rsrc {
        x: x
    }
}
