fn main() {
    ...=.
    //~^ ERROR: unexpected token: `...`
    //~| ERROR: unexpected `=` after inclusive range
    //~| ERROR: expected one of `-`, `/*start of expr expansion*/`, `;`, `}`, or path, found `.`
}
