// pretty-expanded FIXME #23616

struct cat {
  done : extern fn(usize),
  meows : usize,
}

impl Drop for cat {
    fn drop(&mut self) {
        (self.done)(self.meows);
    }
}

fn cat(done: extern fn(usize)) -> cat {
    cat {
        meows: 0,
        done: done
    }
}

pub fn main() {}
