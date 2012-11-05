fn add(n: int) -> fn@(int) -> int {
    fn@(m: int) -> int { m + n }
}

fn main()
{
    assert add(3)(4) == 7;

    let add1 : fn@(int)->int = add(1);
    assert add1(6) == 7;

    let add2 : &(fn@(int)->int) = &add(2);
    assert (*add2)(5) == 7;

    let add3 : fn(int)->int = add(3); //~ ERROR mismatched types
    assert add3(4) == 7;
}
