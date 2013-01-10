extern mod std;

use std::ebml::reader;
use std::ebml::writer;
use std::serialize;

fn main() {
    let foo = ~3;
    let _pfoo = &foo;
    let _f: @fn() -> int = || *foo + 5;
    //~^ ERROR by-move capture

    let bar = ~3;
    let _g = || {
        let _h: @fn() -> int = || *bar;
        //~^ ERROR illegal by-move capture
    };
}
