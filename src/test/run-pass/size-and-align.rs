#![allow(non_camel_case_types)]
enum clam<T> { a(T, isize), b, }

fn uhoh<T>(v: Vec<clam<T>> ) {
    match v[1] {
      clam::a::<T>(ref _t, ref u) => {
          println!("incorrect");
          println!("{}", u);
          panic!();
      }
      clam::b::<T> => { println!("correct"); }
    }
}

pub fn main() {
    let v: Vec<clam<isize>> = vec![clam::b::<isize>, clam::b::<isize>, clam::a::<isize>(42, 17)];
    uhoh::<isize>(v);
}
