use test::{Bencher, black_box};

#[bench]
fn char_iterator(b: &mut Bencher) {
    let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

    b.iter(|| s.chars().count());
}

#[bench]
fn char_iterator_for(b: &mut Bencher) {
    let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

    b.iter(|| {
        for ch in s.chars() { black_box(ch); }
    });
}

#[bench]
fn char_iterator_ascii(b: &mut Bencher) {
    let s = "Mary had a little lamb, Little lamb
    Mary had a little lamb, Little lamb
    Mary had a little lamb, Little lamb
    Mary had a little lamb, Little lamb
    Mary had a little lamb, Little lamb
    Mary had a little lamb, Little lamb";

    b.iter(|| s.chars().count());
}

#[bench]
fn char_iterator_rev(b: &mut Bencher) {
    let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

    b.iter(|| s.chars().rev().count());
}

#[bench]
fn char_iterator_rev_for(b: &mut Bencher) {
    let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

    b.iter(|| {
        for ch in s.chars().rev() { black_box(ch); }
    });
}

#[bench]
fn char_indicesator(b: &mut Bencher) {
    let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
    let len = s.chars().count();

    b.iter(|| assert_eq!(s.char_indices().count(), len));
}

#[bench]
fn char_indicesator_rev(b: &mut Bencher) {
    let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
    let len = s.chars().count();

    b.iter(|| assert_eq!(s.char_indices().rev().count(), len));
}

#[bench]
fn split_unicode_ascii(b: &mut Bencher) {
    let s = "ประเทศไทย中华Việt Namประเทศไทย中华Việt Nam";

    b.iter(|| assert_eq!(s.split('V').count(), 3));
}

#[bench]
fn split_ascii(b: &mut Bencher) {
    let s = "Mary had a little lamb, Little lamb, little-lamb.";
    let len = s.split(' ').count();

    b.iter(|| assert_eq!(s.split(' ').count(), len));
}

#[bench]
fn split_extern_fn(b: &mut Bencher) {
    let s = "Mary had a little lamb, Little lamb, little-lamb.";
    let len = s.split(' ').count();
    fn pred(c: char) -> bool { c == ' ' }

    b.iter(|| assert_eq!(s.split(pred).count(), len));
}

#[bench]
fn split_closure(b: &mut Bencher) {
    let s = "Mary had a little lamb, Little lamb, little-lamb.";
    let len = s.split(' ').count();

    b.iter(|| assert_eq!(s.split(|c: char| c == ' ').count(), len));
}

#[bench]
fn split_slice(b: &mut Bencher) {
    let s = "Mary had a little lamb, Little lamb, little-lamb.";
    let len = s.split(' ').count();

    let c: &[char] = &[' '];
    b.iter(|| assert_eq!(s.split(c).count(), len));
}

#[bench]
fn bench_join(b: &mut Bencher) {
    let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
    let sep = "→";
    let v = vec![s, s, s, s, s, s, s, s, s, s];
    b.iter(|| {
        assert_eq!(v.join(sep).len(), s.len() * 10 + sep.len() * 9);
    })
}

#[bench]
fn bench_contains_short_short(b: &mut Bencher) {
    let haystack = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
    let needle = "sit";

    b.iter(|| {
        assert!(haystack.contains(needle));
    })
}

#[bench]
fn bench_contains_short_long(b: &mut Bencher) {
    let haystack = "\
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse quis lorem sit amet dolor \
ultricies condimentum. Praesent iaculis purus elit, ac malesuada quam malesuada in. Duis sed orci \
eros. Suspendisse sit amet magna mollis, mollis nunc luctus, imperdiet mi. Integer fringilla non \
sem ut lacinia. Fusce varius tortor a risus porttitor hendrerit. Morbi mauris dui, ultricies nec \
tempus vel, gravida nec quam.

In est dui, tincidunt sed tempus interdum, adipiscing laoreet ante. Etiam tempor, tellus quis \
sagittis interdum, nulla purus mattis sem, quis auctor erat odio ac tellus. In nec nunc sit amet \
diam volutpat molestie at sed ipsum. Vestibulum laoreet consequat vulputate. Integer accumsan \
lorem ac dignissim placerat. Suspendisse convallis faucibus lorem. Aliquam erat volutpat. In vel \
eleifend felis. Sed suscipit nulla lorem, sed mollis est sollicitudin et. Nam fermentum egestas \
interdum. Curabitur ut nisi justo.

Sed sollicitudin ipsum tellus, ut condimentum leo eleifend nec. Cras ut velit ante. Phasellus nec \
mollis odio. Mauris molestie erat in arcu mattis, at aliquet dolor vehicula. Quisque malesuada \
lectus sit amet nisi pretium, a condimentum ipsum porta. Morbi at dapibus diam. Praesent egestas \
est sed risus elementum, eu rutrum metus ultrices. Etiam fermentum consectetur magna, id rutrum \
felis accumsan a. Aliquam ut pellentesque libero. Sed mi nulla, lobortis eu tortor id, suscipit \
ultricies neque. Morbi iaculis sit amet risus at iaculis. Praesent eget ligula quis turpis \
feugiat suscipit vel non arcu. Interdum et malesuada fames ac ante ipsum primis in faucibus. \
Aliquam sit amet placerat lorem.

Cras a lacus vel ante posuere elementum. Nunc est leo, bibendum ut facilisis vel, bibendum at \
mauris. Nullam adipiscing diam vel odio ornare, luctus adipiscing mi luctus. Nulla facilisi. \
Mauris adipiscing bibendum neque, quis adipiscing lectus tempus et. Sed feugiat erat et nisl \
lobortis pharetra. Donec vitae erat enim. Nullam sit amet felis et quam lacinia tincidunt. Aliquam \
suscipit dapibus urna. Sed volutpat urna in magna pulvinar volutpat. Phasellus nec tellus ac diam \
cursus accumsan.

Nam lectus enim, dapibus non nisi tempor, consectetur convallis massa. Maecenas eleifend dictum \
feugiat. Etiam quis mauris vel risus luctus mattis a a nunc. Nullam orci quam, imperdiet id \
vehicula in, porttitor ut nibh. Duis sagittis adipiscing nisl vitae congue. Donec mollis risus eu \
leo suscipit, varius porttitor nulla porta. Pellentesque ut sem nec nisi euismod vehicula. Nulla \
malesuada sollicitudin quam eu fermentum.";
    let needle = "english";

    b.iter(|| {
        assert!(!haystack.contains(needle));
    })
}

#[bench]
fn bench_contains_bad_naive(b: &mut Bencher) {
    let haystack = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let needle = "aaaaaaaab";

    b.iter(|| {
        assert!(!haystack.contains(needle));
    })
}

#[bench]
fn bench_contains_equal(b: &mut Bencher) {
    let haystack = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
    let needle = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";

    b.iter(|| {
        assert!(haystack.contains(needle));
    })
}


macro_rules! make_test_inner {
    ($s:ident, $code:expr, $name:ident, $str:expr, $iters:expr) => {
        #[bench]
        fn $name(bencher: &mut Bencher) {
            let mut $s = $str;
            black_box(&mut $s);
            bencher.iter(|| for _ in 0..$iters { black_box($code); });
        }
    }
}

macro_rules! make_test {
    ($name:ident, $s:ident, $code:expr) => {
        make_test!($name, $s, $code, 1);
    };
    ($name:ident, $s:ident, $code:expr, $iters:expr) => {
        mod $name {
            use test::Bencher;
            use test::black_box;

            // Short strings: 65 bytes each
            make_test_inner!($s, $code, short_ascii,
                "Mary had a little lamb, Little lamb Mary had a littl lamb, lamb!", $iters);
            make_test_inner!($s, $code, short_mixed,
                "ศไทย中华Việt Nam; Mary had a little lamb, Little lam!", $iters);
            make_test_inner!($s, $code, short_pile_of_poo,
                "💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩!", $iters);
            make_test_inner!($s, $code, long_lorem_ipsum,"\
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse quis lorem sit amet dolor \
ultricies condimentum. Praesent iaculis purus elit, ac malesuada quam malesuada in. Duis sed orci \
eros. Suspendisse sit amet magna mollis, mollis nunc luctus, imperdiet mi. Integer fringilla non \
sem ut lacinia. Fusce varius tortor a risus porttitor hendrerit. Morbi mauris dui, ultricies nec \
tempus vel, gravida nec quam.

In est dui, tincidunt sed tempus interdum, adipiscing laoreet ante. Etiam tempor, tellus quis \
sagittis interdum, nulla purus mattis sem, quis auctor erat odio ac tellus. In nec nunc sit amet \
diam volutpat molestie at sed ipsum. Vestibulum laoreet consequat vulputate. Integer accumsan \
lorem ac dignissim placerat. Suspendisse convallis faucibus lorem. Aliquam erat volutpat. In vel \
eleifend felis. Sed suscipit nulla lorem, sed mollis est sollicitudin et. Nam fermentum egestas \
interdum. Curabitur ut nisi justo.

Sed sollicitudin ipsum tellus, ut condimentum leo eleifend nec. Cras ut velit ante. Phasellus nec \
mollis odio. Mauris molestie erat in arcu mattis, at aliquet dolor vehicula. Quisque malesuada \
lectus sit amet nisi pretium, a condimentum ipsum porta. Morbi at dapibus diam. Praesent egestas \
est sed risus elementum, eu rutrum metus ultrices. Etiam fermentum consectetur magna, id rutrum \
felis accumsan a. Aliquam ut pellentesque libero. Sed mi nulla, lobortis eu tortor id, suscipit \
ultricies neque. Morbi iaculis sit amet risus at iaculis. Praesent eget ligula quis turpis \
feugiat suscipit vel non arcu. Interdum et malesuada fames ac ante ipsum primis in faucibus. \
Aliquam sit amet placerat lorem.

Cras a lacus vel ante posuere elementum. Nunc est leo, bibendum ut facilisis vel, bibendum at \
mauris. Nullam adipiscing diam vel odio ornare, luctus adipiscing mi luctus. Nulla facilisi. \
Mauris adipiscing bibendum neque, quis adipiscing lectus tempus et. Sed feugiat erat et nisl \
lobortis pharetra. Donec vitae erat enim. Nullam sit amet felis et quam lacinia tincidunt. Aliquam \
suscipit dapibus urna. Sed volutpat urna in magna pulvinar volutpat. Phasellus nec tellus ac diam \
cursus accumsan.

Nam lectus enim, dapibus non nisi tempor, consectetur convallis massa. Maecenas eleifend dictum \
feugiat. Etiam quis mauris vel risus luctus mattis a a nunc. Nullam orci quam, imperdiet id \
vehicula in, porttitor ut nibh. Duis sagittis adipiscing nisl vitae congue. Donec mollis risus eu \
leo suscipit, varius porttitor nulla porta. Pellentesque ut sem nec nisi euismod vehicula. Nulla \
malesuada sollicitudin quam eu fermentum!", $iters);
        }
    }
}

make_test!(chars_count, s, s.chars().count());

make_test!(contains_bang_str, s, s.contains("!"));
make_test!(contains_bang_char, s, s.contains('!'));

make_test!(match_indices_a_str, s, s.match_indices("a").count());

make_test!(split_a_str, s, s.split("a").count());

make_test!(trim_ascii_char, s, {
    s.trim_matches(|c: char| c.is_ascii())
});
make_test!(trim_start_ascii_char, s, {
    s.trim_start_matches(|c: char| c.is_ascii())
});
make_test!(trim_end_ascii_char, s, {
    s.trim_end_matches(|c: char| c.is_ascii())
});

make_test!(find_underscore_char, s, s.find('_'));
make_test!(rfind_underscore_char, s, s.rfind('_'));
make_test!(find_underscore_str, s, s.find("_"));

make_test!(find_zzz_char, s, s.find('\u{1F4A4}'));
make_test!(rfind_zzz_char, s, s.rfind('\u{1F4A4}'));
make_test!(find_zzz_str, s, s.find("\u{1F4A4}"));

make_test!(starts_with_ascii_char, s, s.starts_with('/'), 1024);
make_test!(ends_with_ascii_char, s, s.ends_with('/'), 1024);
make_test!(starts_with_unichar, s, s.starts_with('\u{1F4A4}'), 1024);
make_test!(ends_with_unichar, s, s.ends_with('\u{1F4A4}'), 1024);
make_test!(starts_with_str, s, s.starts_with("💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩"), 1024);
make_test!(ends_with_str, s, s.ends_with("💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩"), 1024);

make_test!(split_space_char, s, s.split(' ').count());
make_test!(split_terminator_space_char, s, s.split_terminator(' ').count());

make_test!(splitn_space_char, s, s.splitn(10, ' ').count());
make_test!(rsplitn_space_char, s, s.rsplitn(10, ' ').count());

make_test!(split_space_str, s, s.split(" ").count());
make_test!(split_ad_str, s, s.split("ad").count());
