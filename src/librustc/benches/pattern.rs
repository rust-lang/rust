use test::Bencher;

// Overhead of various match forms

#[bench]
fn option_some(b: &mut Bencher) {
    let x = Some(10);
    b.iter(|| {
        match x {
            Some(y) => y,
            None => 11
        }
    });
}

#[bench]
fn vec_pattern(b: &mut Bencher) {
    let x = [1,2,3,4,5,6];
    b.iter(|| {
        match x {
            [1,2,3,..] => 10,
            _ => 11,
        }
    });
}
