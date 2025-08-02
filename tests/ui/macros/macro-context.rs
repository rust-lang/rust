// (typeof used because it's surprisingly hard to find an unparsed token after a stmt)
macro_rules! m {
    () => ( i ; typeof );   //~ ERROR expected expression, found reserved keyword `typeof`
                            //~| ERROR macro expansion ignores reserved keyword `typeof`
                            //~| ERROR macro expansion ignores `;`
                            //~| ERROR macro expansion ignores `;`
                            //~| ERROR cannot find type `i` in this scope
                            //~| ERROR cannot find value `i` in this scope
                            //~| ERROR trailing semicolon in macro
                            //~| WARN this was previously
}

fn main() {
    let a: m!();
    let i = m!();
    match 0 {
        m!() => {}
    }

    m!();
}
