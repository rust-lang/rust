//@ check-pass

fn main() {
    let _ = 0111; //~ WARNING [leading_zeros_in_decimal_literals]
    let _ = 0777; //~ WARNING [leading_zeros_in_decimal_literals]
    let _ = 0750; //~ WARNING [leading_zeros_in_decimal_literals]
    let _ = 0007;
    let _ = 0108;
    let _ = 0_0_;
    let _ = 00;
    let _ = 0;
}
