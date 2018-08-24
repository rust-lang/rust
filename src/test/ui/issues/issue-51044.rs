// run-pass
// regression test for issue #50825
// Check that the feature gate normalizes associated types.

#![allow(dead_code)]
struct Foo<T>(T);
struct Duck;
struct Quack;

trait Hello<A> where A: Animal {
}

trait Animal {
    type Noise;
}

trait Loud<R>  {
}

impl Loud<Quack> for f32 {
}

impl Animal for Duck {
    type Noise = Quack;
}

impl Hello<Duck> for Foo<f32> where f32: Loud<<Duck as Animal>::Noise> {
}

fn main() {}
