// compile-flags: -Z parse-only

fn main() {
    let x: [isize 3]; //~ ERROR expected one of `!`, `(`, `+`, `::`, `;`, `<`, or `]`, found `3`
}
