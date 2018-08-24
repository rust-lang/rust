// error-pattern: mismatched types

enum bar { t1((), Option<Vec<isize> >), t2, }

fn foo(t: bar) {
    match t {
      bar::t1(_, Some::<isize>(x)) => {
        println!("{}", x);
      }
      _ => { panic!(); }
    }
}

fn main() { }
