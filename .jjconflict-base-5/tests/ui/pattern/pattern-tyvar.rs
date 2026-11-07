enum Bar { T1((), Option<Vec<isize>>), T2 }

fn foo(t: Bar) {
    match t {
      Bar::T1(_, Some::<isize>(x)) => { //~ ERROR mismatched types
        println!("{}", x);
      }
      _ => { panic!(); }
    }
}

fn main() { }
