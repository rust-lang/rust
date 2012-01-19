// xfail-test

tag a_tag {
    a_tag(u64);
}

type t_rec = {
    c8: u8,
    t: a_tag
};

fn mk_rec() -> t_rec {
    ret { c8:0u8, t:a_tag(0u64) };
}

fn is_8_byte_aligned(&&u: a_tag) -> bool {
    let p = ptr::addr_of(u) as u64;
    ret (p & 7u64) == 0u64;
}

fn main() {
    let x = mk_rec();
    assert is_8_byte_aligned(x.t);
}
