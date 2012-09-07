struct cat {
  done : extern fn(uint),
  meows : uint,
  drop { self.done(self.meows); }
}

fn cat(done: extern fn(uint)) -> cat {
    cat {
        meows: 0u,
        done: done
    }
}

fn main() {}
