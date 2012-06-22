fn main() {
    let a = 1;
    let a_neg: i8 = -a;
    log(error, a_neg);

    let b = 1;
    let b_neg: i16 = -b;
    log(error, b_neg);

    let c = 1;
    let c_neg: i32 = -c;
    log(error, b_neg);

    let d = 1;
    let d_neg: i64 = -d;
    log(error, b_neg);

    let e = 1;
    let e_neg: int = -e;
    log(error, b_neg);

    // intentional overflows

    let f = 1;
    let f_neg: u8 = -f;
    log(error, f_neg);

    let g = 1;
    let g_neg: u16 = -g;
    log(error, g_neg);

    let h = 1;
    let h_neg: u32 = -h;
    log(error, h_neg);

    let i = 1;
    let i_neg: u64 = -i;
    log(error, i_neg);

    let j = 1;
    let j_neg: uint = -j;
    log(error, j_neg);
}
