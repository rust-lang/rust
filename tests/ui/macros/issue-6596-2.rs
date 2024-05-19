macro_rules! g {
    ($inp:ident) => (
        { $inp $nonexistent }
        //~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `$`
    );
}

fn main() {
    let foo = 0;
    g!(foo);
}
