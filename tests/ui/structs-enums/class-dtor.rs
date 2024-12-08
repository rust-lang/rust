//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


struct cat {
  done : extern "C" fn(usize),
  meows : usize,
}

impl Drop for cat {
    fn drop(&mut self) {
        (self.done)(self.meows);
    }
}

fn cat(done: extern "C" fn(usize)) -> cat {
    cat {
        meows: 0,
        done: done
    }
}

pub fn main() {}
