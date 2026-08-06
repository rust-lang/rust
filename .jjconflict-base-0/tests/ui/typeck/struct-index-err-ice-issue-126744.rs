struct X {,} //~ ERROR expected identifier, found `,`

fn main() {
    || {
        if let X { x: 1,} = (X {}) {}
    };
}
