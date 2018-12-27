// compile-flags: -Z parse-only

fn main() {
    for thing(x[]) in foo {} //~ ERROR: expected one of `)`, `,`, or `@`, found `[`
}
