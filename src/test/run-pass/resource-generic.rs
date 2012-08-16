struct finish<T: copy> {
  let arg: {val: T, fin: extern fn(T)};
  new(arg: {val: T, fin: extern fn(T)}) {
    self.arg = arg;
  }
  drop { self.arg.fin(self.arg.val); }
}

fn main() {
    let box = @mut 10;
    fn dec_box(&&i: @mut int) { *i -= 1; }

    { let _i <- finish({val: box, fin: dec_box}); }
    assert (*box == 9);
}
