#![feature(macro_metavar_expr_concat)]

macro_rules! one_rep {
    ( $($a:ident)* ) => {
        $(
            const ${concat($a, Z)}: i32 = 3;
        )*
    };
}

macro_rules! issue_128346 {
    ( $($a:ident)* ) => {
        A(
            const ${concat($a, Z)}: i32 = 3; //~ ERROR `${concat(...)}` variable is still repeating at this depth
        )*
    };
}

macro_rules! issue_131393 {
    ($t:ident $($en:ident)?) => {
        read::<${concat($t, $en)}>()
        //~^ ERROR `${concat(...)}` variable is still repeating at this depth
        //~| ERROR `${concat(...)}` variable is still repeating at this depth
    }
}

fn main() {
    one_rep!(A B C);
    assert_eq!(AZ, 3);
    assert_eq!(BZ, 3);
    assert_eq!(CZ, 3);
    issue_128346!(A B C);
    issue_131393!(u8);
    issue_131393!(u16 le);
}
