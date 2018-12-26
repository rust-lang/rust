// run-pass
// pretty-expanded FIXME #23616

macro_rules! list {
    ( ($($id:ident),*) ) => (());
    ( [$($id:ident),*] ) => (());
    ( {$($id:ident),*} ) => (());
}

macro_rules! tt_list {
    ( ($($tt:tt),*) ) => (());
}

pub fn main() {
    list!( () );
    list!( [] );
    list!( {} );

    tt_list!( (a, b, c) );
    tt_list!( () );
}
