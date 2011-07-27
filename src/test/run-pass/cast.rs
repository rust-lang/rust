


// -*- rust -*-
fn main() {
    let i: int = 'Q' as int;
    assert (i == 0x51);
    let u: u32 = i as u32;
    assert (u == 0x51 as u32);
    assert (u == 'Q' as u32);
    assert (i as u8 == 'Q' as u8);
    assert (i as u8 as i8 == 'Q' as u8 as i8);
    assert (0x51 as char == 'Q');
    assert (true == 1 as bool);
    assert (0 as u32 == false as u32);
}