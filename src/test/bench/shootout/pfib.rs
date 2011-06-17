// -*- rust -*-

/*
  A parallel version of fibonacci numbers.
*/

fn recv[T](&port[T] p) -> T {
  let T x;
  p |> x;
  ret x;
}

fn fib(int n) -> int {
  fn pfib(chan[int] c, int n) {
    if (n == 0) {
      c <| 0;
    }
    else if (n <= 2) {
      c <| 1;
    }
    else {
      let port[int] p = port();
      
      auto t1 = spawn pfib(chan(p), n - 1);
      auto t2 = spawn pfib(chan(p), n - 2);

      c <| recv(p) + recv(p);
    }
  }

  let port[int] p = port();
  auto t = spawn pfib(chan(p), n);
  ret recv(p);
}

fn main() {
    assert (fib(8) == 21);
    assert (fib(15) == 610);
    log fib(8);
    log fib(15);
}
