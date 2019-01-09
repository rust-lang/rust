// run-pass

#![feature(generators)]
#![feature(generator_trait)]

use std::ops::Generator;
use std::ops::GeneratorState;

fn main() {
    let _generator = || {
        let mut sub_generator = || {
            yield 2;
        };

        match unsafe { sub_generator.resume() } {
            GeneratorState::Yielded(x) => {
                yield x;
            }
            _ => panic!(),
        };
    };
}
