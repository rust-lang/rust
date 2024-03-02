struct S;

trait InOut<T> { type Out; }

fn do_fold<B, F: InOut<B, Out=B>>(init: B, f: F) {}

fn bot<T>() -> T { loop {} }

fn main() {
    do_fold(bot(), ()); //~ ERROR trait `InOut<_>` is not implemented for `()`
}
