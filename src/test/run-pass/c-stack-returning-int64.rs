extern mod std;

#[abi = "cdecl"]
#[nolink]
extern mod libc {
    fn atol(x: *u8) -> int;
    fn atoll(x: *u8) -> i64;
}

fn atol(s: ~str) -> int {
    return str::as_buf(s, { |x, _len| libc::atol(x) });
}

fn atoll(s: ~str) -> i64 {
    return str::as_buf(s, { |x, _len| libc::atoll(x) });
}

fn main() {
    assert atol(~"1024") * 10 == atol(~"10240");
    assert (atoll(~"11111111111111111") * 10i64)
        == atoll(~"111111111111111110");
}
