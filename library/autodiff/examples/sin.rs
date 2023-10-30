use autodiff::autodiff;

#[autodiff(cos_inplace, Reverse, Const)]
fn sin_inplace(#[dup] x: &f32, #[dup] y: &mut f32) {
    *y = x.sin();
}


fn main() {
    // Here we can use ==, even though we work on f32.
    // Enzyme will recognize the sin function and replace it with llvm's cos function (see below).
    // Calling f32::cos directly will also result in calling llvm's cos function.
    let a = 3.1415;
    let mut da = 0.0;
    let mut y = 0.0;
    cos_inplace(&a, &mut da, &mut y, &mut 1.0);

    dbg!(&a, &da, &y);
    assert!(da - f32::cos(a) == 0.0);
}

// Just for curious readers, this is the (inner) function that Enzyme does generate:
// define internal { float } @diffe_ZN3sin3sin17h18f17f71fe94e58fE(float %0, float %1) unnamed_addr #35 {
//   %3 = call fast float @llvm.cos.f32(float %0)
//   %4 = fmul fast float %1, %3
//   %5 = insertvalue { float } undef, float %4, 0
//   ret { float } %5
// }

#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main()
    }
}
