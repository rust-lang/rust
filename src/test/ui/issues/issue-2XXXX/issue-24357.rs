struct NoCopy;
fn main() {
   let x = NoCopy;
   //~^ NOTE move occurs because `x` has type `NoCopy`
   let f = move || { let y = x; };
   //~^ NOTE value moved into closure here
   //~| NOTE variable moved due to use in closure
   let z = x;
   //~^ ERROR use of moved value: `x`
   //~| NOTE value used here after move
}
