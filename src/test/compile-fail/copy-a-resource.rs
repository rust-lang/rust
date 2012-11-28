// error-pattern: copying a noncopyable value

struct foo {
  i: int,
}

impl foo : Drop {
    fn finalize(&self) {}
}

fn foo(i:int) -> foo {
    foo {
        i: i
    }
}

fn main() { let x = move foo(10); let y = x; log(error, x); }
