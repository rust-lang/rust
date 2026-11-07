//@ ignore-auxiliary lib.rs
use std::autodiff::autodiff_reverse;
use std::f64::consts::PI;
use std::slice;

fn bitreversal_perm<T>(data: &mut [T]) {
    let len = data.len() / 2;
    let mut j = 1;

    for i in (1..data.len()).step_by(2) {
        if j > i {
            //dbg!(&i, &j);
            data.swap(j - 1, i - 1);
            data.swap(j, i);
            //unsafe {
            //    data.swap_unchecked(j - 1, i - 1);
            //    data.swap_unchecked(j, i);
            //}
        }

        let mut m = len;
        while m >= 2 && j > m {
            j -= m;
            m >>= 1;
        }

        j += m;
    }
}

fn radix2(data: &mut [f64], i_sign: i32) {
    let n = data.len() / 2;
    if n == 1 {
        return;
    }

    let (a, b) = data.split_at_mut(n);
    // assert_eq!(a.len(), b.len());
    radix2(a, i_sign);
    radix2(b, i_sign);

    let wtemp = i_sign as f64 * (PI / n as f64).sin();
    let wpi = -i_sign as f64 * (2.0 * (PI / n as f64)).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let mut wr = 1.0;
    let mut wi = 0.0;

    let (achunks, _) = a.as_chunks_mut();
    let (bchunks, _) = b.as_chunks_mut();
    for ([ax, ay], [bx, by]) in achunks.iter_mut().zip(bchunks.iter_mut()) {
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

fn rescale(data: &mut [f64], scale: usize) {
    let scale = 1. / scale as f64;
    for elm in data {
        *elm *= scale;
    }
}

fn fft(data: &mut [f64]) {
    bitreversal_perm(data);
    radix2(data, 1);
}

fn ifft(data: &mut [f64]) {
    bitreversal_perm(data);
    radix2(data, -1);
    rescale(data, data.len() / 2);
}

#[autodiff_reverse(dfoobar, DuplicatedOnly)]
pub fn foobar(data: &mut [f64]) {
    fft(data);
    ifft(data);
}

#[no_mangle]
pub extern "C" fn rust_dfoobar(n: usize, data: *mut f64, ddata: *mut f64) {
    let (data, ddata) = unsafe {
        (
            slice::from_raw_parts_mut(data, n * 2),
            slice::from_raw_parts_mut(ddata, n * 2),
        )
    };

    unsafe { dfoobar(data, ddata) };
}

#[no_mangle]
pub extern "C" fn rust_foobar(n: usize, data: *mut f64) {
    let data = unsafe { slice::from_raw_parts_mut(data, n * 2) };
    foobar(data);
}
