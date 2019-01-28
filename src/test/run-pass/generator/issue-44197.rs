// run-pass

#![feature(generators, generator_trait)]

use std::ops::{ Generator, GeneratorState };
use std::pin::Pin;

fn foo(_: &str) -> String {
    String::new()
}

fn bar(baz: String) -> impl Generator<Yield = String, Return = ()> {
    move || {
        yield foo(&baz);
    }
}

fn foo2(_: &str) -> Result<String, ()> {
    Err(())
}

fn bar2(baz: String) -> impl Generator<Yield = String, Return = ()> {
    move || {
        if let Ok(quux) = foo2(&baz) {
            yield quux;
        }
    }
}

fn main() {
    assert_eq!(Pin::new(&mut bar(String::new())).resume(), GeneratorState::Yielded(String::new()));
    assert_eq!(Pin::new(&mut bar2(String::new())).resume(), GeneratorState::Complete(()));
}
