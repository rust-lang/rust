type t<T> = { f: fn() -> T };

fn f<T>(_x: t<T>) {}

fn main() {
  let x: t<()> = { f: || () }; //~ ERROR expressions with stack closure
    f(x);
}
