resource shrinky_pointer(i: @mutable int) { *i -= 1; }

fn look_at(pt: shrinky_pointer) -> int { ret **pt; }

fn main() {
    let my_total = @mutable 10;
    { let pt <- shrinky_pointer(my_total); assert (look_at(pt) == 10); }
    assert (*my_total == 9);
}
