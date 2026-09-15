// An argument that fails to parse for an unrelated reason must not make us blame an earlier
// comparison for a missing turbofish.

fn take_three(_: bool, _: bool, _: ()) {}

fn take_generic<T>(_: bool, _: T, _: ()) {}

struct S;
struct Many<A, B, C, D>(A, B, C, D);
impl<A, B, C, D> Many<A, B, C, D> {
    fn new() -> Self {
        todo!()
    }
}

fn main() {
    let (a, b, c, d) = (1, 2, 3, 4);
    take_three(a < b, c > (d), @);
    //~^ ERROR expected expression, found `@`
}

fn closure_argument() {
    let (a, b) = (1, 2);
    take_generic(a < b, || 0, @);
    //~^ ERROR expected expression, found `@`
}

fn missing_comma() {
    let (a, b, c, d) = (1, 2, 3, 4);
    take_three(a < b, c d, @);
    //~^ ERROR expected one of `!`, `)`, `,`, `.`, `::`, `?`, `{`, or an operator, found `d`
    //~| ERROR expected expression, found `@`
}

fn non_path_lhs() {
    take_generic(1 < 2, Many<(), i32, S, S>, i32, i32>::new(), ());
    //~^ ERROR expected expression, found `,`
}
