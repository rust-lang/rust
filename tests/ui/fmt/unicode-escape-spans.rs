fn main() {
    // 1 byte in UTF-8
    format!("\u{000041}{a}"); //~ ERROR cannot find value
    format!("\u{0041}{a}"); //~ ERROR cannot find value
    format!("\u{41}{a}"); //~ ERROR cannot find value
    format!("\u{0}{a}"); //~ ERROR cannot find value

    // 2 bytes
    format!("\u{0df}{a}"); //~ ERROR cannot find value
    format!("\u{df}{a}"); //~ ERROR cannot find value

    // 3 bytes
    format!("\u{00211d}{a}"); //~ ERROR cannot find value
    format!("\u{211d}{a}"); //~ ERROR cannot find value

    // 4 bytes
    format!("\u{1f4a3}{a}"); //~ ERROR cannot find value
    format!("\u{10ffff}{a}"); //~ ERROR cannot find value
}
