// compile-flags: -Z parse-only

fn main() {
    let t = (42, 42);
    t.0::<isize>; //~ ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `::`
}
