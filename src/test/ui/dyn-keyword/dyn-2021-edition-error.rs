// edition:2021
fn function(x: &SomeTrait, y: Box<SomeTrait>) {
    //~^ ERROR trait objects without an explicit `dyn` are deprecated
    //~| ERROR trait objects without an explicit `dyn` are deprecated
    let _x: &SomeTrait = todo!();
    //~^ ERROR trait objects without an explicit `dyn` are deprecated
}

trait SomeTrait {}

fn main() {}
