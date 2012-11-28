// error-pattern: copying a noncopyable value

struct r {
  i: @mut int,
}

impl r : Drop {
    fn finalize(&self) {
        *(self.i) = *(self.i) + 1;
    }
}

fn f<T>(+i: ~[T], +j: ~[T]) {
    let k = i + j;
}

fn main() {
    let i1 = @mut 0;
    let i2 = @mut 1;
    let r1 = move ~[~r { i: i1 }];
    let r2 = move ~[~r { i: i2 }];
    f(r1, r2);
    log(debug, (r2, *i1));
    log(debug, (r1, *i2));
}
