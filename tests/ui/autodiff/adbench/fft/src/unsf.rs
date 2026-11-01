//@ ignore-auxiliary lib.rs
use std::autodiff::autodiff_reverse;
use std::f64::consts::PI;

unsafe fn bitreversal_perm(data: *mut f64, len: usize) {
    let mut j = 1;

    for i in (1..2 * len).step_by(2) {
        if j > i {
            std::ptr::swap(data.add(j - 1), data.add(i - 1));
            std::ptr::swap(data.add(j), data.add(i));
        }

        let mut m = len;
        while m >= 2 && j > m {
            j -= m;
            m >>= 1;
        }

        j += m;
    }
}

unsafe fn radix2(data: *mut f64, n: usize, i_sign: i32) {
    if n == 1 {
        return;
    }
    radix2(data, n / 2, i_sign);
    radix2(data.add(n), n / 2, i_sign);

    let wtemp = i_sign as f64 * (PI / n as f64).sin();
    let wpi = -i_sign as f64 * (2.0 * (PI / n as f64)).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let mut wr = 1.0;
    let mut wi = 0.0;

    for i in (0..n).step_by(2) {
        let in_n = i + n;
        let ax = &mut *data.add(i);
        let ay = &mut *data.add(i + 1);
        let bx = &mut *data.add(in_n);
        let by = &mut *data.add(in_n + 1);
        let tempr = *bx * wr - *by * wi;
        let tempi = *bx * wi + *by * wr;

        *bx = *ax - tempr;
        *by = *ay - tempi;
        *ax += tempr;
        *ay += tempi;

        let wtemp_new = wr;
        wr = wr * (wpr + 1.0) - wi * wpi;
        wi = wi * (wpr + 1.0) + wtemp_new * wpi;
    }
}

unsafe fn rescale(data: *mut f64, n: usize) {
    let scale = 1. / n as f64;
    for i in 0..2 * n {
        *data.add(i) = *data.add(i) * scale;
    }
}

unsafe fn fft(data: *mut f64, n: usize) {
    bitreversal_perm(data, n);
    radix2(data, n, 1);
}

unsafe fn ifft(data: *mut f64, n: usize) {
    bitreversal_perm(data, n);
    radix2(data, n, -1);
    rescale(data, n);
}

#[autodiff_reverse(unsafe_dfoobar, Const, DuplicatedOnly)]
pub unsafe fn unsafe_foobar(n: usize, data: *mut f64) {
    fft(data, n);
    ifft(data, n);
}

#[no_mangle]
pub extern "C" fn rust_unsafe_dfoobar(n: usize, data: *mut f64, ddata: *mut f64) {
    unsafe {
        unsafe_dfoobar(n, data, ddata);
    }
}

#[no_mangle]
pub extern "C" fn rust_unsafe_foobar(n: usize, data: *mut f64) {
    unsafe {
        unsafe_foobar(n, data);
    }
}
