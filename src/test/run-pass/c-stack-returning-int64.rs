use std;
import str;

#[abi = "cdecl"]
#[link_name = ""]
native mod libc {
    fn atol(x: str::sbuf) -> int;
    fn atoll(x: str::sbuf) -> i64;
}

fn atol(s: str) -> int {
    ret str::as_buf(s, { |x| libc::atol(x) });
}

fn atoll(s: str) -> i64 {
    ret str::as_buf(s, { |x| libc::atoll(x) });
}

fn main() {
    assert atol("1024") * 10 == atol("10240");
    assert (atoll("11111111111111111") * 10i64)
        == atoll("111111111111111110");
}
