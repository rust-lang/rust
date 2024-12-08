struct NoCopy; //~ NOTE if `NoCopy` implemented `Clone`, you could clone the value
//~^ NOTE consider implementing `Clone` for this type
fn main() {
   let x = NoCopy;
   //~^ NOTE move occurs because `x` has type `NoCopy`
   let f = move || { let y = x; };
   //~^ NOTE value moved into closure here
   //~| NOTE variable moved due to use in closure
   //~| NOTE you could clone this value
   let z = x;
   //~^ ERROR use of moved value: `x`
   //~| NOTE value used here after move
}
