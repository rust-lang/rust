enum opts {
    a(int), b(int), c(int)
}

fn matcher1(x: opts) {
    match x {
      a(ref i) | b(copy i) => {} //~ ERROR variable `i` is bound with different mode in pattern #2 than in pattern #1
      c(_) => {}
    }
}

fn matcher2(x: opts) {
    match x {
      a(ref i) | b(i) => {} //~ ERROR variable `i` is bound with different mode in pattern #2 than in pattern #1
      c(_) => {}
    }
}

fn matcher3(x: opts) {
    match x {
      a(ref mut i) | b(ref const i) => {} //~ ERROR variable `i` is bound with different mode in pattern #2 than in pattern #1
      c(_) => {}
    }
}

fn matcher4(x: opts) {
    match x {
      a(ref mut i) | b(ref i) => {} //~ ERROR variable `i` is bound with different mode in pattern #2 than in pattern #1
      c(_) => {}
    }
}


fn matcher5(x: opts) {
    match x {
      a(ref i) | b(ref i) => {}
      c(_) => {}
    }
}

fn main() {}
