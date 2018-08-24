// compile-flags: -Z parse-only -Z continue-parse-after-error


pub fn main() {
    br"é";  //~ ERROR raw byte string must be ASCII
    br##~"a"~##;  //~ ERROR only `#` is allowed in raw string delimitation
}
