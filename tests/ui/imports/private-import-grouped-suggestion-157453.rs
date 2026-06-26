mod one {
    pub struct One();
}

mod two {
    use crate::one::One;
    pub struct Two();
}

mod test_grouped {
    use crate::two::{One, Two}; //~ ERROR struct import `One` is private [E0603]
}

mod test_single_item {
    use crate::two::{One}; //~ ERROR struct import `One` is private [E0603]
}

mod outer {
    pub mod inner {
        pub struct MyPath;
    }
}

mod reexport {
    use crate::outer::inner::MyPath;
}

mod test_std_style {
    use crate::reexport::{MyPath}; //~ ERROR struct import `MyPath` is private [E0603]
}

fn main() {}
