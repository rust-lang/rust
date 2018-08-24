fn main() {
    let Test(&desc[..]) = x; //~ ERROR: expected one of `)`, `,`, or `@`, found `[`
}
