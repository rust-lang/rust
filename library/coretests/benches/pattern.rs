use test::{Bencher, black_box};

#[bench]
fn starts_with_char(b: &mut Bencher) {
    let text = black_box("kdjsfhlakfhlsghlkvcnljknfqiunvcijqenwodind");
    b.iter(|| {
        for _ in 0..1024 {
            black_box(text.starts_with('k'));
        }
    })
}

#[bench]
fn starts_with_str(b: &mut Bencher) {
    let text = black_box("kdjsfhlakfhlsghlkvcnljknfqiunvcijqenwodind");
    b.iter(|| {
        for _ in 0..1024 {
            black_box(text.starts_with("k"));
        }
    })
}

#[bench]
fn ends_with_char(b: &mut Bencher) {
    let text = black_box("kdjsfhlakfhlsghlkvcnljknfqiunvcijqenwodind");
    b.iter(|| {
        for _ in 0..1024 {
            black_box(text.ends_with('k'));
        }
    })
}

#[bench]
fn ends_with_str(b: &mut Bencher) {
    let text = black_box("kdjsfhlakfhlsghlkvcnljknfqiunvcijqenwodind");
    b.iter(|| {
        for _ in 0..1024 {
            black_box(text.ends_with("k"));
        }
    })
}

fn make_haystack() -> String {
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse quis lorem \
    sit amet dolor ultricies condimentum. Praesent iaculis purus elit, ac malesuada \
    quam malesuada in. Duis sed orci eros. Suspendisse sit amet magna mollis, mollis \
    nunc luctus, imperdiet mi. Integer fringilla non sem ut lacinia. Fusce varius \
    tortor a risus porttitor hendrerit. Morbi mauris dui, ultricies nec tempus vel, \
    gravida nec quam. In est dui, tincidunt sed tempus interdum, adipiscing laoreet \
    ante. Etiam tempor, tellus quis sagittis interdum, nulla purus mattis sem, quis \
    auctor erat odio ac tellus. In nec nunc sit amet diam volutpat molestie at sed \
    ipsum. Vestibulum laoreet consequat vulputate. Integer accumsan lorem ac dignissim \
    placerat. Suspendisse convallis faucibus lorem. Aliquam erat volutpat."
        .repeat(50)
}

#[bench]
fn find_str(b: &mut Bencher) {
    let s = make_haystack();
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.find("the english language")))
}

#[bench]
fn rfind_str(b: &mut Bencher) {
    let s = make_haystack();
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.rfind("the english language")))
}

#[bench]
fn find_str_worst_case(b: &mut Bencher) {
    let near_miss = "the english languagX";
    let haystack_str = near_miss.repeat(2000);
    let haystack = black_box(haystack_str.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.find("the english language")))
}

#[bench]
fn rfind_str_worst_case(b: &mut Bencher) {
    let near_miss = "the english languagX";
    let haystack_str = near_miss.repeat(2000);
    let haystack = black_box(haystack_str.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.rfind("the english language")))
}
