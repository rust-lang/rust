//@ check-pass
macro_rules! bad {
    ($s:ident whatever) => {
        {
            let $s = 0;
            *&mut $s = 0;
            //~^ WARNING cannot borrow `foo` as mutable, as it is not declared as mutable [E0596]
        }
    }
}

fn main() {
    bad!(foo whatever);
}
