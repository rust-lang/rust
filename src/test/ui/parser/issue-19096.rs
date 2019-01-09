fn main() {
    let t = (42, 42);
    t.0::<isize>; //~ ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `::`
                  //~| ERROR mismatched types
}
