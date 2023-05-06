#![rustfmt::skip]

mod a {
    mod b {
        mod c {
            mod d {
                mod e {}
            }
        }
    }
}

fn main() {
    // this should lint
    {{{}}}
}
