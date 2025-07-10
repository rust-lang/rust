mod a {
    use std::marker::PhantomData;

    enum Bug {
        V = [PhantomData; { [ () ].len() ].len() as isize,
    }
}

mod b {
    enum Bug {
        V = [Vec::new; { [].len()  ].len() as isize,
    }
}

mod c {
    enum Bug {
        V = [Vec::new; { [0].len() ].len() as isize,
}

fn main() {} //~ ERROR this file contains an unclosed delimiter
