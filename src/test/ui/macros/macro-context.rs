// (typeof used because it's surprisingly hard to find an unparsed token after a stmt)
macro_rules! m {
    () => ( i ; typeof );   //~ ERROR expected expression, found reserved keyword `typeof`
                            //~| ERROR macro expansion ignores token `typeof`
                            //~| ERROR macro expansion ignores token `;`
                            //~| ERROR macro expansion ignores token `;`
}

fn main() {
    let a: m!();
    let i = m!();
    match 0 {
        m!() => {}
    }

    m!();
}
