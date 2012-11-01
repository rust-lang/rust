// xfail-test
enum side { mayo, catsup, vinegar }
enum order { hamburger, fries(side), shake }
enum meal { to_go(order), for_here(order) }

fn foo(m: @meal, cond: bool) {
    match *m {
      to_go(_) => { }
      for_here(_) if cond => {}
      for_here(hamburger) => {}
      for_here(fries(_s)) => {}
      for_here(shake) => {}
    }
}

fn main() {
    foo(@for_here(hamburger), true)
}
