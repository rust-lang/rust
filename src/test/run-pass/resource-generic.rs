// xfail-fast
#[legacy_modes];

struct finish<T: Copy> {
  arg: {val: T, fin: extern fn(T)},
}

impl<T: Copy> finish<T> : Drop {
    fn finalize(&self) {
        self.arg.fin(self.arg.val);
    }
}

fn finish<T: Copy>(arg: {val: T, fin: extern fn(T)}) -> finish<T> {
    finish {
        arg: arg
    }
}

fn main() {
    let box = @mut 10;
    fn dec_box(&&i: @mut int) { *i -= 1; }

    { let _i = move finish({val: box, fin: dec_box}); }
    assert (*box == 9);
}
