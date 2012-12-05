fn main() {
    let x = ~{mut a: ~10, b: ~20};
    match x {
      ~{ref a, ref b} => { assert **a == 10; (*x).a = ~30; assert **a == 30; }
    }
}
