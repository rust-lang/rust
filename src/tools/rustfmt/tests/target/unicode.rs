// rustfmt-wrap_comments: true

fn foo() {
    let s = "this line goes to 100: ͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶͶ";
    let s = 42;

    // a comment of length 80, with the starting sigil: ҘҘҘҘҘҘҘҘҘҘ ҘҘҘҘҘҘҘҘҘҘҘҘҘҘ
    let s = 42;
}

pub fn bar(config: &Config) {
    let csv = RefCell::new(create_csv(config, "foo"));
    {
        let mut csv = csv.borrow_mut();
        for (i1, i2, i3) in iproduct!(0..2, 0..3, 0..3) {
            csv.write_field(format!("γ[{}.{}.{}]", i1, i2, i3)).unwrap();
            csv.write_field(format!("d[{}.{}.{}]", i1, i2, i3)).unwrap();
            csv.write_field(format!("i[{}.{}.{}]", i1, i2, i3)).unwrap();
        }
        csv.write_record(None::<&[u8]>).unwrap();
    }
}

// The NotUnicode line is below 100 wrt chars but over it wrt String::len
fn baz() {
    let our_error_b = result_b_from_func.or_else(|e| match e {
        NotPresent => Err(e).chain_err(|| "env var wasn't provided"),
        NotUnicode(_) => Err(e).chain_err(|| "env var was very very very borkæ–‡å­—åŒ–ã"),
    });
}
