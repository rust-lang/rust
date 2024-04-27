// rustfmt-indent_style: Visual
// Function arguments layout

fn lorem() {}

fn lorem(ipsum: usize) {}

fn lorem(ipsum: usize,
         dolor: usize,
         sit: usize,
         amet: usize,
         consectetur: usize,
         adipiscing: usize,
         elit: usize) {
    // body
}

// #1922
extern "C" {
    pub fn LAPACKE_csytrs_rook_work(matrix_layout: c_int,
                                    uplo: c_char,
                                    n: lapack_int,
                                    nrhs: lapack_int,
                                    a: *const lapack_complex_float,
                                    lda: lapack_int,
                                    ipiv: *const lapack_int,
                                    b: *mut lapack_complex_float,
                                    ldb: lapack_int)
                                    -> lapack_int;

    pub fn LAPACKE_csytrs_rook_work(matrix_layout: c_int,
                                    uplo: c_char,
                                    n: lapack_int,
                                    nrhs: lapack_int,
                                    lda: lapack_int,
                                    ipiv: *const lapack_int,
                                    b: *mut lapack_complex_float,
                                    ldb: lapack_int)
                                    -> lapack_int;
}
