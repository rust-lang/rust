//@ dont-require-annotations: NOTE

fn main() {
    unsafe {
        dealloc(ptr2, Layout::(x: !)(1, 1)); //~ ERROR: expected one of `!`, `(`, `)`, `+`, `,`, `::`, or `<`, found `:`
        //~^ ERROR: expected one of `.`, `;`, `?`, `}`, or an operator, found `)`
        //~| NOTE while parsing this parenthesized list of type arguments starting here
    }
}
