use test::Bencher;

#[bench]
fn bench_str_under_8_bytes_eq(b: &mut Bencher) {
    let s = "foo";
    let other = "FOo";
    b.iter(|| {
        assert!(s.eq_ignore_ascii_case(other));
    })
}

#[bench]
fn bench_str_of_8_bytes_eq(b: &mut Bencher) {
    let s = "foobar78";
    let other = "FOObAr78";
    b.iter(|| {
        assert!(s.eq_ignore_ascii_case(other));
    })
}

#[bench]
fn bench_str_17_bytes_eq(b: &mut Bencher) {
    let s = "performance-criti";
    let other = "performANce-cRIti";
    b.iter(|| {
        assert!(s.eq_ignore_ascii_case(other));
    })
}

#[bench]
fn bench_str_31_bytes_eq(b: &mut Bencher) {
    let s = "foobarbazquux02foobarbazquux025";
    let other = "fooBARbazQuuX02fooBARbazQuuX025";
    b.iter(|| {
        assert!(s.eq_ignore_ascii_case(other));
    })
}

#[bench]
fn bench_long_str_eq(b: &mut Bencher) {
    let s = "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor \
             incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud \
             exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute \
             irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla \
             pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui \
             officia deserunt mollit anim id est laborum.";
    let other = "Lorem ipsum dolor sit amet, CONSECTETUR adipisicing elit, sed do eiusmod tempor \
             incididunt ut labore et dolore MAGNA aliqua. Ut enim ad MINIM veniam, quis nostrud \
             exercitation ullamco LABORIS nisi ut aliquip ex ea commodo consequat. Duis aute \
             irure dolor in reprehenderit in voluptate velit esse cillum DOLORE eu fugiat nulla \
             pariatur. Excepteur sint occaecat CUPIDATAT non proident, sunt in culpa qui \
             officia deserunt mollit anim id est laborum.";
    b.iter(|| {
        assert!(s.eq_ignore_ascii_case(other));
    })
}
