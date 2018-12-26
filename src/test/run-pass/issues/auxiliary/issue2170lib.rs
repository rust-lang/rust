fn foo(_x: i32) {
}

pub struct rsrc {
  x: i32,
}

impl Drop for rsrc {
    fn drop(&mut self) {
        foo(self.x);
    }
}

pub fn rsrc(x: i32) -> rsrc {
    rsrc {
        x: x
    }
}
