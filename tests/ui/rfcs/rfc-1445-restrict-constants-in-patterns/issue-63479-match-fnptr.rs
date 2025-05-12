// The actual regression test from #63479. (Including this because my
// first draft at fn-ptr-is-structurally-matchable.rs failed to actually
// cover the case this hit; I've since expanded it accordingly, but the
// experience left me wary of leaving this regression test out.)

#[derive(Eq)]
struct A {
  a: i64
}

impl PartialEq for A {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.a.eq(&other.a)
    }
}

type Fn = fn(&[A]);

fn my_fn(_args: &[A]) {
  println!("hello world");
}

const TEST: Fn = my_fn;
const TEST2: (Fn, u8) = (TEST, 0);

struct B(Fn);

fn main() {
  let s = B(my_fn);
  match s {
    B(TEST) => println!("matched"),
    //~^ ERROR behave unpredictably
    _ => panic!("didn't match")
  };
  match (s.0, 0) {
    TEST2 => println!("matched"),
    //~^ ERROR behave unpredictably
    _ => panic!("didn't match")
  }
}
