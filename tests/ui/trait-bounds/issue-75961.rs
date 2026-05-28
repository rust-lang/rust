//@ check-pass

pub fn foo<'a>(s: &'a mut ()) where &'a mut (): Clone {
    <&mut () as Clone>::clone(&s);
}

fn main() {}
