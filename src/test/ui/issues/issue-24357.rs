struct NoCopy;
fn main() {
   let x = NoCopy;
   let f = move || { let y = x; };
   //~^ NOTE value moved (into closure) here
   let z = x;
   //~^ ERROR use of moved value: `x`
   //~| NOTE value used here after move
   //~| NOTE move occurs because `x` has type `NoCopy`
}
