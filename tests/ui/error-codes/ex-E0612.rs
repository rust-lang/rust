struct Foo(u32);

fn main() {
   let y = Foo(0);
   y.1; //~ ERROR no field `1` on type `Foo`
}
