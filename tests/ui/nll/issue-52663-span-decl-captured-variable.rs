fn expect_fn<F>(f: F) where F : Fn() {
    f();
}

fn main() {
   {
       let x = (vec![22], vec![44]);
       expect_fn(|| drop(x.0));
       //~^ ERROR cannot move out of `x.0`, as `x` is a captured variable in an `Fn` closure [E0507]
   }
}
