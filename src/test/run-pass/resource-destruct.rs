resource shrinky_pointer(@mutable int i) {
    *i -= 1;
}

fn look_at(&shrinky_pointer pt) -> int {
    ret **pt;
}

fn main() {
    auto my_total = @mutable 10;
    {
        auto pt <- shrinky_pointer(my_total);
        assert (look_at(pt) == 10);
    }
    assert (*my_total == 9);
}
