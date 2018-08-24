pub fn main() {
    'a: { //~ ERROR labels on blocks are unstable
        break 'a;
    }
}
