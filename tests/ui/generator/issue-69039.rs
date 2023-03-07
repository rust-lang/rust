// run-pass

#![feature(generators, generator_trait)]

use std::ops::{Generator, GeneratorState};

fn mkstr(my_name: String, my_mood: String) -> String {
    format!("{} is {}", my_name.trim(), my_mood.trim())
}

fn my_scenario() -> impl Generator<String, Yield = &'static str, Return = String> {
    |_arg: String| {
        let my_name = yield "What is your name?";
        let my_mood = yield "How are you feeling?";
        mkstr(my_name, my_mood)
    }
}

fn main() {
    let mut my_session = Box::pin(my_scenario());

    assert_eq!(
        my_session.as_mut().resume("_arg".to_string()),
        GeneratorState::Yielded("What is your name?")
    );
    assert_eq!(
        my_session.as_mut().resume("Your Name".to_string()),
        GeneratorState::Yielded("How are you feeling?")
    );
    assert_eq!(
        my_session.as_mut().resume("Sensory Organs".to_string()),
        GeneratorState::Complete("Your Name is Sensory Organs".to_string())
    );
}
