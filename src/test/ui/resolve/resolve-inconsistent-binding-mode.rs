enum opts {
    a(isize), b(isize), c(isize)
}

fn matcher1(x: opts) {
    match x {
      opts::a(ref i) | opts::b(i) => {}
      //~^ ERROR variable `i` is bound in inconsistent ways within the same match arm
      //~^^ ERROR mismatched types
      opts::c(_) => {}
    }
}

fn matcher2(x: opts) {
    match x {
      opts::a(ref i) | opts::b(i) => {}
      //~^ ERROR variable `i` is bound in inconsistent ways within the same match arm
      //~^^ ERROR mismatched types
      opts::c(_) => {}
    }
}

fn matcher4(x: opts) {
    match x {
      opts::a(ref mut i) | opts::b(ref i) => {}
      //~^ ERROR variable `i` is bound in inconsistent ways within the same match arm
      //~^^ ERROR mismatched types
      opts::c(_) => {}
    }
}


fn matcher5(x: opts) {
    match x {
      opts::a(ref i) | opts::b(ref i) => {}
      opts::c(_) => {}
    }
}

fn main() {}
