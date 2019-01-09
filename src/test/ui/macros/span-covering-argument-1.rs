macro_rules! bad {
    ($s:ident whatever) => {
        {
            let $s = 0;
            *&mut $s = 0;
            //~^ ERROR cannot borrow immutable local variable `foo` as mutable [E0596]
        }
    }
}

fn main() {
    bad!(foo whatever);
}
