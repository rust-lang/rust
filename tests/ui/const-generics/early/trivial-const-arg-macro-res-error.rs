// This is a regression test for #128016.

macro_rules! len {
    () => {
        target
        //~^ ERROR cannot find value `target`
    };
}

fn main() {
    let val: [str; len!()] = [];
    //~^ ERROR the size for values
}
