#![allow(unused_macros)]

macro_rules! errors_everywhere {
    ($ty:ty <) => (); //~ ERROR `$ty:ty` is followed by `<`, which is not allowed for `ty`
    ($ty:ty < foo ,) => (); //~ ERROR `$ty:ty` is followed by `<`, which is not allowed for `ty`
    ($ty:ty , ) => ();
    ( ( $ty:ty ) ) => ();
    ( { $ty:ty } ) => ();
    ( [ $ty:ty ] ) => ();
    ($bl:block < ) => ();
    ($pa:pat >) => (); //~ ERROR `$pa:pat` is followed by `>`, which is not allowed for `pat`
    ($pa:pat , ) => ();
    ($pa:pat $pb:pat $ty:ty ,) => ();
    //~^ ERROR `$pa:pat` is followed by `$pb:pat`, which is not allowed
    //~^^ ERROR `$pb:pat` is followed by `$ty:ty`, which is not allowed
    ($($ty:ty)* -) => (); //~ ERROR `$ty:ty` is followed by `-`
    ($($a:ty, $b:ty)* -) => (); //~ ERROR `$b:ty` is followed by `-`
    ($($ty:ty)-+) => (); //~ ERROR `$ty:ty` is followed by `-`, which is not allowed for `ty`
    ( $($a:expr)* $($b:tt)* ) => { };
    //~^ ERROR `$a:expr` is followed by `$b:tt`, which is not allowed for `expr` fragments
}

fn main() { }
