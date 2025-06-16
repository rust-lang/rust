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

#[bench]
fn splitn_on_http_response(b: &mut Bencher) {
    fn parse_http(s: &str) -> Result<(&str, &str, &str), &str> {
        let mut parts = s.splitn(3, ' ');
        let version = parts.next().ok_or("No version")?;
        let code = parts.next().ok_or("No status code")?;
        let description = parts.next().ok_or("No description")?;
        Ok((version, code, description))
    }

    let response = String::from("HTTP/1.1 418 I'm a teapot\r\n");
    let mut res: (&str, &str, &str) = ("", "", "");
    b.iter(|| {
        for _ in 0..1024 {
            res = black_box(match parse_http(black_box(&response)) {
                Ok(data) => data,
                Err(_) => {
                    continue;
                }
            })
        }
    })
}
