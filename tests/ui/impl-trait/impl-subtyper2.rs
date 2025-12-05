//@ check-pass

fn ages() -> Option<impl Iterator> {
    None::<std::slice::Iter<()>>
}

fn main(){}
