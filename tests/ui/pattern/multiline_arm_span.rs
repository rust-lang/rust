#![deny(unreachable_patterns)]

fn main(){
    let x = 42;
    match x{
        73 => {}
        irrefutable => {}
        //~^ this pattern is irrefutable; subsequent arms are never executed
        other_irrefutable => {
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
            // a big and beautiful multi line match arm
            todo!()
        }
    }
}
