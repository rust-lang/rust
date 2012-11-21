fn main() {
    let v = @mut [ 1, 2, 3 ];
    for v.each |_x| {   //~ ERROR illegal borrow
        v[1] = 4;
    }
}

