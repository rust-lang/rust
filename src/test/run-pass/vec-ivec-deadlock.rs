// xfail-stage0

fn main() {
    auto a = ~[ 1, 2, 3, 4, 5 ];
    auto b = [ a, a ];
    b += b;
}
