// run-pass
macro_rules! print_hd_tl {
    ($field_hd:ident, $($field_tl:ident),+) => ({
        print!("{}", stringify!($field_hd));
        print!("::[");
        $(
            print!("{}", stringify!($field_tl));
            print!(", ");
        )+
        print!("]\n");
    })
}

pub fn main() {
    print_hd_tl!(x, y, z, w)
}
