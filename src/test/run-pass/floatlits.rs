

fn main() {
    auto f = 4.999999999999;
    assert (f > 4.90);
    assert (f < 5.0);
    auto g = 4.90000000001e-10;
    assert (g > 5e-11);
    assert (g < 5e-9);
}