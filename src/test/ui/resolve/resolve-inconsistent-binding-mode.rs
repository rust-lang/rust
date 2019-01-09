enum Opts {
    A(isize), B(isize), C(isize)
}

fn matcher1(x: Opts) {
    match x {
      Opts::A(ref i) | Opts::B(i) => {}
      //~^ ERROR variable `i` is bound in inconsistent ways within the same match arm
      //~^^ ERROR mismatched types
      Opts::C(_) => {}
    }
}

fn matcher2(x: Opts) {
    match x {
      Opts::A(ref i) | Opts::B(i) => {}
      //~^ ERROR variable `i` is bound in inconsistent ways within the same match arm
      //~^^ ERROR mismatched types
      Opts::C(_) => {}
    }
}

fn matcher4(x: Opts) {
    match x {
      Opts::A(ref mut i) | Opts::B(ref i) => {}
      //~^ ERROR variable `i` is bound in inconsistent ways within the same match arm
      //~^^ ERROR mismatched types
      Opts::C(_) => {}
    }
}


fn matcher5(x: Opts) {
    match x {
      Opts::A(ref i) | Opts::B(ref i) => {}
      Opts::C(_) => {}
    }
}

fn main() {}
