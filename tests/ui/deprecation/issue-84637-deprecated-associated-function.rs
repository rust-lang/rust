// run-rustfix

#![deny(deprecated)]

fn main() {
    let _foo = str::trim_left("   aoeu"); //~ ERROR use of deprecated method `core::str::<impl str>::trim_left`: superseded by `trim_start` [deprecated]

    let _bar = "   aoeu".trim_left(); //~ ERROR use of deprecated method `core::str::<impl str>::trim_left`: superseded by `trim_start` [deprecated]
}
