#![feature(macro_metavar_expr)]
#![feature(macro_metavar_expr_concat)]

macro_rules! one_rep {
    ( $($a:ident)* ) => {
        $(
            const ${concat($a, Z)}: i32 = 3;
        )*
    };
}

macro_rules! issue_127723 {
    ($($a:ident, $c:ident;)*) => {
        $(
            const ${concat($a, B, $c, D)}: i32 = 3;
        )*
    };
}

macro_rules! issue_127723_ignore {
    ($($a:ident, $c:ident;)*) => {
        $(
            ${ignore($a)}
            ${ignore($c)}
            const ${concat($a, B, $c, D)}: i32 = 3;
        )*
    };
}

macro_rules! issue_128346 {
    ( $($a:ident)* ) => {
        A(
            const ${concat($a, Z)}: i32 = 3;
            //~^ ERROR invalid syntax
        )*
    };
}

macro_rules! issue_131393 {
    ($t:ident $($en:ident)?) => {
        read::<${concat($t, $en)}>();
        //~^ ERROR invalid syntax
        //~| ERROR invalid syntax
    }
}

fn main() {
    one_rep!(A B C);
    assert_eq!(AZ, 3);
    assert_eq!(BZ, 3);
    assert_eq!(CZ, 3);
    issue_127723! {
        A, D;
        B, E;
        C, F;
    }
    issue_127723_ignore! {
        N, O;
        P, Q;
        R, S;
    }
    issue_128346!(A B C);
    issue_131393!(u8);
    issue_131393!(u16 le);
}
