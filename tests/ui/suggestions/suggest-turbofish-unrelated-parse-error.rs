// An argument that fails to parse for an unrelated reason must not make us blame an earlier
// comparison for a missing turbofish.

fn take_three(_: bool, _: bool, _: ()) {}

fn main() {
    let (a, b, c, d) = (1, 2, 3, 4);
    take_three(a < b, c > (d), @);
    //~^ ERROR expected expression, found `@`
}
