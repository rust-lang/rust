// error-pattern: copying a noncopyable value

// Test that a class with a non-copyable field can't be
// copied
struct bar {
  x: int,
}

impl bar : Drop {
    fn finalize(&self) {}
}

fn bar(x:int) -> bar {
    bar {
        x: x
    }
}

struct foo {
  i: int,
  j: bar,
}

fn foo(i:int) -> foo {
    foo {
        i: i,
        j: bar(5)
    }
}

fn main() { let x = move foo(10); let y = copy x; log(error, x); }
