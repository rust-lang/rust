fn main() {
    unsafe {
        dealloc(ptr2, Layout::(x: !)(1, 1)); //~ ERROR: expected one of `!`, `(`, `)`, `+`, `,`, `::`, or `<`, found `:`
        //~^ ERROR: expected one of `.`, `;`, `?`, `}`, or an operator, found `)`
        //~| while parsing this parenthesized list of type arguments starting here
    }
}
