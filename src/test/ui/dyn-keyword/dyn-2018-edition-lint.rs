// edition:2018
#[deny(bare_trait_objects)]

fn function(x: &SomeTrait, y: Box<SomeTrait>) {
    //~^ ERROR trait objects without an explicit `dyn` are deprecated
    //~| WARN this was previously accepted
    //~| ERROR trait objects without an explicit `dyn` are deprecated
    //~| WARN this was previously accepted
    let _x: &SomeTrait = todo!();
    //~^ ERROR trait objects without an explicit `dyn` are deprecated
    //~| WARN this was previously accepted
}

trait SomeTrait {}

fn main() {}
