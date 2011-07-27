// -*- rust -*-

// Tests of ports and channels on various types
fn test_rec() {
    type r = {val0: int, val1: u8, val2: char};

    let po: port[r] = port();
    let ch: chan[r] = chan(po);
    let r0: r = {val0: 0, val1: 1u8, val2: '2'};
    ch <| r0;
    let r1: r;
    po |> r1;
    assert (r1.val0 == 0);
    assert (r1.val1 == 1u8);
    assert (r1.val2 == '2');
}

fn test_vec() {
    let po: port[int[]] = port();
    let ch: chan[int[]] = chan(po);
    let v0: int[] = ~[0, 1, 2];
    ch <| v0;
    let v1: int[];
    po |> v1;
    assert (v1.(0) == 0);
    assert (v1.(1) == 1);
    assert (v1.(2) == 2);
}

fn test_str() {
    let po: port[str] = port();
    let ch: chan[str] = chan(po);
    let s0: str = "test";
    ch <| s0;
    let s1: str;
    po |> s1;
    assert (s1.(0) as u8 == 't' as u8);
    assert (s1.(1) as u8 == 'e' as u8);
    assert (s1.(2) as u8 == 's' as u8);
    assert (s1.(3) as u8 == 't' as u8);
}

fn test_tag() {
    tag t { tag1; tag2(int); tag3(int, u8, char); }
    let po: port[t] = port();
    let ch: chan[t] = chan(po);
    ch <| tag1;
    ch <| tag2(10);
    ch <| tag3(10, 11u8, 'A');
    let t1: t;
    po |> t1;
    assert (t1 == tag1);
    po |> t1;
    assert (t1 == tag2(10));
    po |> t1;
    assert (t1 == tag3(10, 11u8, 'A'));
}

fn test_chan() {
    let po: port[chan[int]] = port();
    let ch: chan[chan[int]] = chan(po);
    let po0: port[int] = port();
    let ch0: chan[int] = chan(po0);
    ch <| ch0;
    let ch1: chan[int];
    po |> ch1;
    // Does the transmitted channel still work?

    ch1 <| 10;
    let i: int;
    po0 |> i;
    assert (i == 10);
}

fn main() { test_rec(); test_vec(); test_str(); test_tag(); test_chan(); }