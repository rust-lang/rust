// ignore-tidy-cr

pub fn main() {
    br"a"; //~ ERROR bare CR not allowed in raw string
    br"é";  //~ ERROR raw byte string must be ASCII
    br##~"a"~##;  //~ ERROR only `#` is allowed in raw string delimitation
}
