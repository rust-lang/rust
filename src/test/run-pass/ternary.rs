fn test_simple() { let x = true ? 10 : 11; assert (x == 10); }

fn test_precedence() {
    let x;

    x = true || true ? 10 : 11;
    assert (x == 10);

    x = true == false ? 10 : 11;
    assert (x == 11);

    x = true ? false ? 10 : 11 : 12;
    assert (x == 11);

    let y = true ? 0xF0 : 0x0 | 0x0F;
    assert (y == 0xF0);

    y = true ? 0xF0 | 0x0F : 0x0;
    assert (y == 0xFF);
}

fn test_associativity() {
    // Ternary is right-associative
    let x = false ? 10 : false ? 11 : 12;
    assert (x == 12);
}

fn test_lval() {
    let box1: @mutable int = @mutable 10;
    let box2: @mutable int = @mutable 10;
    *(true ? box1 : box2) = 100;
    assert (*box1 == 100);
}

fn test_as_stmt() { let s; true ? s = 10 : s = 12; assert (s == 10); }

fn main() {
    test_simple();
    test_precedence();
    test_associativity();
    test_lval();
    test_as_stmt();
}