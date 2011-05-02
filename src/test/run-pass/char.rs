fn main() {
    let char c = 'x';
    let char d = 'x';
    check(c == 'x');
    check('x' == c);
    check(c == c);
    check(c == d);
    check(d == c);
    check (d == 'x');
    check('x' == d);
}

