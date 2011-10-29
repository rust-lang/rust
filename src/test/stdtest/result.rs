import std::result;

fn op1() -> result::t<int, str> { result::ok(666) }

fn op2(&&i: int) -> result::t<uint, str> { result::ok(i as uint + 1u) }

fn op3() -> result::t<int, str> { result::err("sadface") }

#[test]
fn chain_success() {
    assert result::get(result::chain(op1(), op2)) == 667u;
}

#[test]
fn chain_failure() {
    assert result::get_err(result::chain(op3(), op2)) == "sadface";
}