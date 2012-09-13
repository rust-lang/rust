enum color {
    red = 0xff0000,
    green = 0x00ff00,
    blue = 0x0000ff,
    black = 0x000000,
    white = 0xFFFFFF,
}

fn main() {
    let act = fmt!("%?", red);
    io::println(act);
    assert ~"red" == act;
    assert ~"green" == fmt!("%?", green);
    assert ~"white" == fmt!("%?", white);
}

