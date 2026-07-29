// Regression test for issue #54771.

trait Bar {}

impl Bar for u8 {} //~ HELP the trait `Bar` is implemented for `u8`

fn bar<R: Bar>(_: impl Fn() -> R) {}

fn main() {
    bar(|| { //~ ERROR the trait bound `(): Bar` is not satisfied
        5u8; //~ HELP remove this semicolon
    });

    bar(|| { //~ ERROR the trait bound `(): Bar` is not satisfied
        5u8;
        ()
    });
}
